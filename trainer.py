import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import Config, get_config
from model import DualHeadModel, load_teacher_model
from data_utils import create_dataloaders, prepare_batch_for_generation


class TrainingMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.student_entropies = []
        self.teacher_entropies = []
        self.num_switches = 0
        self.num_tokens_generated = 0
        self.training_steps = 0

    def update(self, training_logs: List[Dict], num_tokens: int):
        for log in training_logs:
            self.losses.append(log['loss'])
            self.student_entropies.append(log['student_entropy'])
            self.teacher_entropies.append(log['teacher_entropy'])
            self.training_steps += 1
        self.num_switches += len(training_logs)
        self.num_tokens_generated += num_tokens

    def get_summary(self) -> Dict:
        if not self.losses:
            return {"avg_loss": 0.0, "num_switches": 0, "num_tokens": 0, "training_steps": 0}
        return {
            "avg_loss": np.mean(self.losses),
            "std_loss": np.std(self.losses),
            "avg_student_entropy": np.mean(self.student_entropies),
            "avg_teacher_entropy": np.mean(self.teacher_entropies),
            "num_switches": self.num_switches,
            "num_tokens": self.num_tokens_generated,
            "training_steps": self.training_steps,
            "switches_per_token": self.num_switches / max(1, self.num_tokens_generated)
        }


class DualHeadTrainer:
    def __init__(self, model: DualHeadModel, teacher_model: nn.Module, tokenizer: Any, config: Config, output_dir: Optional[str] = None):
        self.model = model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.config = config

        self.output_dir = Path(output_dir or config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        self.optimizer = self._create_optimizer()
        self.scheduler = None
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "timestamps": []
        }

    def _create_optimizer(self) -> optim.Optimizer:
        trainable_params = self.model.get_trainable_parameters()
        return optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def _create_scheduler(self, num_training_steps: int) -> Any:
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.config.training.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_training_steps - self.config.training.warmup_steps, eta_min=1e-6
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.config.training.warmup_steps]
        )

    def train_on_batch(self, batch: Dict, verbose: bool = False) -> Tuple[Dict, List[str]]:
        sequences = prepare_batch_for_generation(
            batch, self.tokenizer, next(self.model.base_model.parameters()).device
        )

        batch_metrics = TrainingMetrics()
        generated_texts = []

        for i, input_ids in enumerate(sequences):
            try:
                generated_ids, training_logs = self.model.dynamic_generate_and_train(
                    input_ids=input_ids,
                    teacher_model=self.teacher_model,
                    optimizer=self.optimizer,
                    max_new_tokens=self.config.inference.max_new_tokens,
                    temperature=self.config.inference.temperature,
                    top_p=self.config.inference.top_p,
                    verbose=verbose
                )
                num_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
                batch_metrics.update(training_logs, num_new_tokens)
                generated_text = self.tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)
                generated_texts.append(generated_text)
            except Exception as e:
                continue

        return batch_metrics.get_summary(), generated_texts

    def validate(self, val_loader: Any, num_batches: Optional[int] = None) -> Dict:
        self.model.eval()
        self.val_metrics.reset()
        num_batches = num_batches or len(val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break
                sequences = prepare_batch_for_generation(
                    batch, self.tokenizer, next(self.model.base_model.parameters()).device
                )
                batch_logs = []
                for input_ids in sequences:
                    try:
                        is_switch = self.model.switch_detector.is_switch_position(input_ids.squeeze(0))
                        if is_switch:
                            loss, student_probs, teacher_probs = self.model.compute_distillation_loss(
                                input_ids, self.teacher_model, self.config.training.distill_temperature
                            )
                            log_entry = {
                                'loss': loss.item(),
                                'student_entropy': -torch.sum(student_probs * torch.log(student_probs + 1e-8)).item(),
                                'teacher_entropy': -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8)).item()
                            }
                            batch_logs.append(log_entry)
                    except Exception:
                        continue
                if batch_logs:
                    self.val_metrics.update(batch_logs, 0)

        return self.val_metrics.get_summary()

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "head_1_state_dict": self.model.head_1.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config.to_dict()
        }
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

        head1_path = self.checkpoint_dir / f"head_1_epoch_{epoch}.pt"
        self.model.save_head_1(str(head1_path))
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.config.training.save_total_limit:
            for checkpoint in checkpoints[:-self.config.training.save_total_limit]:
                checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.head_1.load_state_dict(checkpoint["head_1_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        return checkpoint["epoch"]

    def save_training_history(self):
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def log_metrics(self, epoch: int, batch_idx: int, batch_metrics: Dict, mode: str = "train"):
        self.history[f"{mode}_losses"].append(batch_metrics.get("avg_loss", 0))
        self.history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
        self.history["timestamps"].append(time.time())

        if self.global_step % self.config.training.save_steps == 0:
            metrics_path = self.log_dir / f"metrics_step_{self.global_step}.json"
            with open(metrics_path, 'w') as f:
                json.dump(batch_metrics, f, indent=2)

    def train(self, train_loader: Any, val_loader: Optional[Any] = None, num_epochs: Optional[int] = None, resume_from: Optional[str] = None):
        num_epochs = num_epochs or self.config.training.num_epochs
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        total_steps = num_epochs * len(train_loader)
        self.scheduler = self._create_scheduler(total_steps)

        try:
            for epoch in range(start_epoch, num_epochs):
                self.model.train()
                self.train_metrics.reset()
                epoch_texts = []

                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
                for batch_idx, batch in enumerate(progress_bar):
                    batch_metrics, generated_texts = self.train_on_batch(batch, verbose=False)
                    self.train_metrics.losses.extend([batch_metrics['avg_loss']] if batch_metrics['avg_loss'] > 0 else [])
                    epoch_texts.extend(generated_texts)

                    progress_bar.set_postfix({
                        'loss': f"{batch_metrics['avg_loss']:.4f}",
                        'switches': batch_metrics['num_switches'],
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

                    self.log_metrics(epoch, batch_idx, batch_metrics)

                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1

                    if val_loader and self.global_step % self.config.training.eval_steps == 0:
                        val_metrics = self.validate(val_loader, num_batches=5)
                        self.log_metrics(epoch, batch_idx, val_metrics, mode="val")
                        if val_metrics['avg_loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['avg_loss']
                            self.save_checkpoint(epoch, val_metrics, is_best=True)

                    if self.global_step % self.config.training.save_steps == 0:
                        epoch_summary = self.train_metrics.get_summary()
                        self.save_checkpoint(epoch, epoch_summary)

                if val_loader:
                    val_metrics = self.validate(val_loader)
                    if val_metrics['avg_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['avg_loss']
                        self.save_checkpoint(epoch, val_metrics, is_best=True)

                if epoch_texts:
                    samples_path = self.log_dir / f"samples_epoch_{epoch+1}.txt"
                    with open(samples_path, 'w') as f:
                        for i, text in enumerate(epoch_texts[:5]):
                            f.write(f"Sample {i+1}:\n{text}\n\n{'='*50}\n\n")

                self.save_training_history()

        except KeyboardInterrupt:
            self.save_checkpoint(epoch, self.train_metrics.get_summary())
            self.save_training_history()

        except Exception as e:
            self.save_checkpoint(epoch, self.train_metrics.get_summary())
            raise


def create_trainer(model: DualHeadModel, teacher_model: nn.Module, tokenizer: Any, config: Optional[Config] = None, output_dir: Optional[str] = None) -> DualHeadTrainer:
    if config is None:
        config = get_config()
    return DualHeadTrainer(model=model, teacher_model=teacher_model, tokenizer=tokenizer, config=config, output_dir=output_dir)
