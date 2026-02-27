import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from config import Config, get_config


@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    head_0_logits: Optional[torch.Tensor] = None
    head_1_logits: Optional[torch.Tensor] = None
    head_1_probs: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    is_switch_position: bool = False
    loss: Optional[torch.Tensor] = None


class SpecialVocabMapper:
    def __init__(self, tokenizer: AutoTokenizer, special_vocab: List[str]):
        self.tokenizer = tokenizer
        self.special_vocab = special_vocab
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        self.vocab_ids = []
        self._build_mapping()

    def _build_mapping(self):
        for i, word in enumerate(self.special_vocab):
            strategies = [word, f" {word}", f"{word} ", f" {word} "]
            token_id = None
            for strategy in strategies:
                token_ids = self.tokenizer.encode(strategy, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_id = token_ids[0]
                    break
                elif len(token_ids) > 0 and token_id is None:
                    token_id = token_ids[0]

            if token_id is not None:
                self.vocab_to_id[word] = token_id
                self.id_to_vocab[token_id] = word
                self.vocab_ids.append(token_id)
            else:
                raise ValueError(f"Cannot tokenize special word: {word}")

        self.vocab_ids = torch.tensor(self.vocab_ids, dtype=torch.long)

    def get_vocab_ids(self) -> torch.Tensor:
        return self.vocab_ids

    def sample_from_head1_logits(self, head1_logits: torch.Tensor, temperature: float = 1.0) -> int:
        probs = F.softmax(head1_logits / temperature, dim=-1)
        vocab_ids_on_device = self.vocab_ids.to(head1_logits.device)
        idx = torch.multinomial(probs, 1).item()
        return vocab_ids_on_device[idx].item()


class SwitchDetector:
    def __init__(self, tokenizer: AutoTokenizer, switch_delimiter: str = "\n\n"):
        self.tokenizer = tokenizer
        self.switch_delimiter = switch_delimiter

    def is_switch_position(self, input_ids: torch.Tensor) -> bool:
        if len(input_ids) < 3:
            return False
        try:
            last_tokens = input_ids[-1:].tolist()
            decoded_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
            return decoded_text.endswith(self.switch_delimiter)
        except Exception:
            return False


class DualHeadModel(nn.Module):
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.slm_model_path,
            torch_dtype=config.model.torch_dtype,
            device_map=config.model.device_map
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        config.model.hidden_size = self.hidden_size

        self.head_0 = self.base_model.lm_head
        self.vocab_mapper = SpecialVocabMapper(tokenizer, config.model.special_vocab)
        self.switch_detector = SwitchDetector(tokenizer, config.model.switch_delimiter)

        self.head_1 = nn.Linear(self.hidden_size, config.model.head_1_output_dim, bias=False)
        self._initialize_head_1_from_lm_head()

        base_device = next(self.base_model.parameters()).device
        base_dtype = next(self.base_model.parameters()).dtype
        self.head_1 = self.head_1.to(device=base_device, dtype=base_dtype)

        self.tpv_weights = None
        self.tpv_enabled = False

    def load_tpv_model(self, tpv_weights_path: str):
        tpv_path = Path(tpv_weights_path)
        if not tpv_path.exists():
            raise FileNotFoundError(f"TPV weights not found: {tpv_weights_path}")

        weights_np = np.load(tpv_weights_path)
        model_device = next(self.base_model.parameters()).device
        model_dtype = next(self.base_model.parameters()).dtype

        self.tpv_weights = torch.tensor(weights_np, dtype=model_dtype, device=model_device)

        if self.tpv_weights.shape[0] != self.hidden_size:
            raise ValueError(
                f"TPV weights dimension {self.tpv_weights.shape[0]} "
                f"doesn't match model hidden size {self.hidden_size}"
            )
        self.tpv_enabled = True

    def predict_tpv_score(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if not self.tpv_enabled or self.tpv_weights is None:
            return torch.tensor(0.0, device=hidden_state.device)
        hidden_state = hidden_state.to(self.tpv_weights.device)
        return torch.dot(self.tpv_weights, hidden_state)

    def predict_tpv_scores_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.tpv_enabled or self.tpv_weights is None:
            return torch.zeros(hidden_states.shape[0], device=hidden_states.device)
        hidden_states = hidden_states.to(self.tpv_weights.device)
        return hidden_states @ self.tpv_weights

    def compute_thinking_token_scores(
            self,
            input_ids: torch.Tensor,
            head_1_logits: torch.Tensor,
            temperature: float = 0.6
    ) -> torch.Tensor:
        final_scores = head_1_logits.clone()
        if not self.tpv_enabled:
            return final_scores

        vocab_ids = self.vocab_mapper.get_vocab_ids()
        num_tokens = vocab_ids.shape[0]

        with torch.no_grad():
            prefix_outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True
            )
            past_key_values = prefix_outputs.past_key_values
            past_key_values.batch_repeat_interleave(num_tokens)

            candidate_input = vocab_ids.to(
                device=input_ids.device, dtype=input_ids.dtype
            ).unsqueeze(1)

            candidate_outputs = self.base_model(
                input_ids=candidate_input,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )

            last_hidden_batch = candidate_outputs.hidden_states[-1][:, -1, :]
            del prefix_outputs, past_key_values, candidate_outputs

        length_scores = self.predict_tpv_scores_batch(last_hidden_batch)
        final_scores = final_scores + length_scores.to(final_scores.device, dtype=final_scores.dtype)
        return final_scores

    def _initialize_head_1_from_lm_head(self):
        vocab_ids = self.vocab_mapper.get_vocab_ids()
        lm_head_weight = self.head_0.weight.data
        special_token_weights = lm_head_weight[vocab_ids, :]
        with torch.no_grad():
            self.head_1.weight.data = special_token_weights.clone()

    def _ensure_input_device(self, input_ids: torch.Tensor) -> torch.Tensor:
        model_device = next(self.base_model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        return input_ids

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            use_head_1: bool = False
    ) -> Union[ModelOutput, Tuple]:
        input_ids = self._ensure_input_device(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = base_outputs.hidden_states[-1]
        last_hidden = hidden_states[:, -1, :]
        head_0_logits = self.head_0(last_hidden)

        head_1_logits = None
        head_1_probs = None

        if use_head_1:
            head_1_device = next(self.head_1.parameters()).device
            last_hidden_on_device = last_hidden.to(head_1_device)
            head_1_logits = self.head_1(last_hidden_on_device)
            head_1_probs = F.softmax(head_1_logits, dim=-1)

        is_switch = self.switch_detector.is_switch_position(input_ids.squeeze(0))

        if return_dict:
            return ModelOutput(
                logits=head_0_logits,
                head_0_logits=head_0_logits,
                head_1_logits=head_1_logits,
                head_1_probs=head_1_probs,
                hidden_states=last_hidden,
                is_switch_position=is_switch
            )
        else:
            return (head_0_logits,)

    def generate_next_token(
            self,
            input_ids: torch.Tensor,
            temperature: float = 0.6,
            top_p: float = 0.95,
            use_head_1: bool = False
    ) -> int:
        with torch.no_grad():
            outputs = self.forward(input_ids, use_head_1=use_head_1)

            if use_head_1 and outputs.head_1_logits is not None:
                head_1_logits = outputs.head_1_logits.squeeze(0)
                return self.vocab_mapper.sample_from_head1_logits(head_1_logits, temperature)
            else:
                logits = outputs.head_0_logits.squeeze(0) / temperature
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()

    def compute_distillation_loss(
            self,
            input_ids: torch.Tensor,
            teacher_model: nn.Module,
            temperature: float = 4.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_outputs = self.forward(input_ids, use_head_1=True)
        student_logits = student_outputs.head_1_logits.squeeze(0) / temperature

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, return_dict=True)
            teacher_full_logits = teacher_outputs.logits[:, -1, :]
            vocab_ids = self.vocab_mapper.get_vocab_ids().to(teacher_full_logits.device)
            teacher_special_logits = teacher_full_logits.squeeze(0)[vocab_ids] / temperature

        target_device = student_logits.device
        teacher_special_logits = teacher_special_logits.to(target_device)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_special_logits, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss, student_probs, teacher_probs

    def dynamic_generate_and_train(
            self,
            input_ids: torch.Tensor,
            teacher_model: nn.Module,
            optimizer: torch.optim.Optimizer,
            max_new_tokens: int = 100,
            temperature: float = 0.6,
            top_p: float = 0.95,
            verbose: bool = False
    ) -> Tuple[torch.Tensor, List[Dict]]:
        generated_ids = input_ids.clone()
        training_logs = []
        self.eval()

        for step in range(max_new_tokens):
            is_switch = self.switch_detector.is_switch_position(generated_ids.squeeze(0))

            if is_switch:
                self.head_1.train()
                optimizer.zero_grad()

                try:
                    loss, student_probs, teacher_probs = self.compute_distillation_loss(
                        generated_ids, teacher_model, self.config.training.distill_temperature
                    )

                    loss.backward()

                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.get_trainable_parameters(),
                            self.config.training.max_grad_norm
                        )

                    optimizer.step()

                    log_entry = {
                        'step': step,
                        'loss': loss.item(),
                        'student_entropy': -torch.sum(student_probs * torch.log(student_probs + 1e-8)).item(),
                        'teacher_entropy': -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8)).item(),
                        'max_student_prob': torch.max(student_probs).item(),
                        'max_teacher_prob': torch.max(teacher_probs).item()
                    }
                    training_logs.append(log_entry)

                except Exception as e:
                    pass

                self.eval()

                try:
                    next_token = self.generate_next_token(
                        generated_ids, temperature, top_p, use_head_1=True
                    )
                except Exception:
                    next_token = self.generate_next_token(
                        generated_ids, temperature, top_p, use_head_1=False
                    )
            else:
                next_token = self.generate_next_token(
                    generated_ids, temperature, top_p, use_head_1=False
                )

            new_token_tensor = torch.tensor([[next_token]],
                                            device=generated_ids.device,
                                            dtype=generated_ids.dtype)
            generated_ids = torch.cat([generated_ids, new_token_tensor], dim=1)

            if next_token == self.tokenizer.eos_token_id:
                break

        return generated_ids, training_logs

    def get_trainable_parameters(self):
        return list(self.head_1.parameters())

    def save_head_1(self, path: str):
        torch.save(self.head_1.state_dict(), path)

    def load_head_1(self, path: str):
        self.head_1.load_state_dict(torch.load(path))


def create_dual_head_model(config: Optional[Config] = None) -> Tuple[DualHeadModel, AutoTokenizer]:
    if config is None:
        config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config.model.slm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.inference.pad_token_id = tokenizer.pad_token_id
    model = DualHeadModel(config, tokenizer)
    return model, tokenizer


def load_teacher_model(config: Config) -> AutoModelForCausalLM:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.model.llm_model_path,
        torch_dtype=config.model.torch_dtype,
        device_map=config.model.device_map
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    return teacher_model
