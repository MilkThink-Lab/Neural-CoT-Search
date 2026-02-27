import argparse
import sys
import os
from pathlib import Path
import torch
from transformers import set_seed, AutoTokenizer
import json
from datetime import datetime

sys.path.append('.')

from config import Config, get_config, update_config
from model import create_dual_head_model, load_teacher_model
from data_utils import create_dataloaders, analyze_dataset
from trainer import create_trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Dual-Head Distillation Model")
    parser.add_argument("--slm_model_path", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def setup_config(args) -> Config:
    config = get_config()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        for section_name, section_values in custom_config.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_values.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    if args.slm_model_path:
        config.model.slm_model_path = args.slm_model_path
    if args.llm_model_path:
        config.model.llm_model_path = args.llm_model_path
    if args.dataset_name:
        config.data.dataset_name = args.dataset_name
    if args.num_samples is not None:
        config.data.num_samples = args.num_samples
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.max_new_tokens is not None:
        config.inference.max_new_tokens = args.max_new_tokens
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.experiment_name:
        config.system.experiment_name = args.experiment_name
    if args.seed is not None:
        config.system.seed = args.seed
    if args.num_gpus is not None:
        config.system.num_gpus = args.num_gpus

    if args.debug:
        config.data.num_samples = 10
        config.training.num_epochs = 1
        config.inference.max_new_tokens = 100
        config.training.logging_steps = 1
        config.training.eval_steps = 5

    return config


def main():
    args = parse_arguments()
    config = setup_config(args)
    set_seed(config.system.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.training.output_dir) / f"{config.system.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = create_dual_head_model(config)
        teacher_model = load_teacher_model(config)
        train_loader, val_loader = create_dataloaders(config, tokenizer)

        trainer = create_trainer(
            model=model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            config=config,
            output_dir=str(output_dir)
        )

        if not args.debug:
            response = input("Proceed with training? (y/n): ")
            if response.lower() != 'y':
                return

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            resume_from=args.resume_from
        )

        summary = {
            "config": config.to_dict(),
            "final_metrics": {
                "train_loss": trainer.history["train_losses"][-1] if trainer.history["train_losses"] else None,
                "val_loss": trainer.history["val_losses"][-1] if trainer.history["val_losses"] else None,
                "best_val_loss": trainer.best_val_loss,
                "total_steps": trainer.global_step
            },
            "output_dir": str(output_dir)
        }

        summary_path = output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
