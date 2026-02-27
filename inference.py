import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import time
import json
from dataclasses import dataclass, field

from config import Config, get_config
from model import DualHeadModel, create_dual_head_model


@dataclass
class GenerationOutput:
    text: str
    tokens: List[int] = field(default_factory=list)
    switch_positions: List[int] = field(default_factory=list)
    head_1_tokens: List[int] = field(default_factory=list)
    generation_time: float = 0.0
    num_tokens: int = 0
    tokens_per_second: float = 0.0
    metadata: Dict = field(default_factory=dict)


class DualHeadInference:
    def __init__(self, model: DualHeadModel, tokenizer, config: Optional[Config] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or get_config()
        self.model.eval()
        self.device = next(self.model.base_model.parameters()).device

    def load_checkpoint(self, checkpoint_path: Union[str, Path], load_head_1_only: bool = True):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".pt":
            if checkpoint_path.name.startswith("head_1"):
                self.model.load_head_1(str(checkpoint_path))
            else:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "head_1_state_dict" in checkpoint:
                    self.model.head_1.load_state_dict(checkpoint["head_1_state_dict"])
                else:
                    self.model.head_1.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")

    def load_tpv_model(self, tpv_weights_path: str):
        self.model.load_tpv_model(tpv_weights_path)

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            max_new_tokens: Optional[int] = None,
            max_paragraphs: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            do_sample: Optional[bool] = None,
            use_head_1_at_switch: Optional[bool] = None,
            head_1_temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            verbose: bool = False
    ) -> GenerationOutput:
        max_new_tokens = max_new_tokens or self.config.inference.max_new_tokens
        temperature = temperature or self.config.inference.temperature
        top_p = top_p or self.config.inference.top_p
        do_sample = do_sample if do_sample is not None else self.config.inference.do_sample
        use_head_1_at_switch = use_head_1_at_switch if use_head_1_at_switch is not None else self.config.inference.use_head_1_at_switch
        head_1_temperature = head_1_temperature or self.config.inference.head_1_temperature

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        initial_length = input_ids.shape[1]

        output = GenerationOutput(text="", metadata={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "max_paragraphs": max_paragraphs,
            "temperature": temperature,
            "use_head_1_at_switch": use_head_1_at_switch,
            "tpv_enabled": self.model.tpv_enabled
        })

        start_time = time.time()
        generated_ids = input_ids.clone()
        paragraph_count = 0
        accumulated_text = ""

        for step in range(max_new_tokens):
            is_switch = self.model.switch_detector.is_switch_position(generated_ids.squeeze(0))
            use_head_1 = is_switch and use_head_1_at_switch
            model_outputs = self.model.forward(generated_ids, use_head_1=use_head_1)

            if use_head_1 and model_outputs.head_1_logits is not None:
                head_1_logits = model_outputs.head_1_logits.squeeze(0)
                final_scores = self.model.compute_thinking_token_scores(
                    input_ids=generated_ids,
                    head_1_logits=head_1_logits,
                    temperature=head_1_temperature
                )
                scaled_scores = final_scores / head_1_temperature

                if do_sample:
                    probs = F.softmax(scaled_scores, dim=-1)
                    vocab_ids = self.model.vocab_mapper.get_vocab_ids().to(self.device)
                    idx = torch.multinomial(probs, 1).item()
                    next_token = vocab_ids[idx].item()
                else:
                    vocab_ids = self.model.vocab_mapper.get_vocab_ids().to(self.device)
                    idx = torch.argmax(scaled_scores).item()
                    next_token = vocab_ids[idx].item()

                output.head_1_tokens.append(next_token)
                output.switch_positions.append(step)
            else:
                logits = model_outputs.head_0_logits.squeeze(0)
                if do_sample:
                    logits = logits / temperature
                    if 0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            0, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = -float('inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = torch.argmax(logits).item()

            output.tokens.append(next_token)
            new_token_tensor = torch.tensor([[next_token]], device=self.device, dtype=generated_ids.dtype)
            generated_ids = torch.cat([generated_ids, new_token_tensor], dim=1)

            if max_paragraphs is not None:
                token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                accumulated_text += token_text
                if "\n\n" in accumulated_text:
                    new_paragraph_count = accumulated_text.count("\n\n")
                    paragraph_count += new_paragraph_count
                    if accumulated_text.endswith("\n"):
                        accumulated_text = "\n"
                    else:
                        accumulated_text = ""
                    if paragraph_count >= max_paragraphs:
                        output.metadata["early_stop_reason"] = "max_paragraphs_reached"
                        output.metadata["paragraph_count"] = paragraph_count
                        break

            if next_token == self.tokenizer.eos_token_id:
                break

            if stop_sequences:
                current_text = self.tokenizer.decode(
                    generated_ids.squeeze(0)[initial_length:],
                    skip_special_tokens=True
                )
                if any(seq in current_text for seq in stop_sequences):
                    break

        output.text = self.tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)
        output.generation_time = time.time() - start_time
        output.num_tokens = len(output.tokens)
        output.tokens_per_second = output.num_tokens / output.generation_time if output.generation_time > 0 else 0

        if max_paragraphs is not None:
            final_paragraph_count = output.text.count("\n\n")
            output.metadata["final_paragraph_count"] = final_paragraph_count

        return output

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], **generation_kwargs) -> List[GenerationOutput]:
        outputs = []
        for prompt in prompts:
            output = self.generate(prompt, **generation_kwargs)
            outputs.append(output)
        return outputs

    def benchmark(self, prompt: str, num_runs: int = 5, **generation_kwargs) -> Dict:
        times = []
        tokens_generated = []
        head_1_usage = []

        for i in range(num_runs):
            output = self.generate(prompt, verbose=False, **generation_kwargs)
            times.append(output.generation_time)
            tokens_generated.append(output.num_tokens)
            head_1_usage.append(len(output.head_1_tokens))

        results = {
            "num_runs": num_runs,
            "avg_time": sum(times) / num_runs,
            "std_time": torch.tensor(times).std().item() if num_runs > 1 else 0,
            "avg_tokens": sum(tokens_generated) / num_runs,
            "avg_tokens_per_second": sum(t / time for t, time in zip(tokens_generated, times)) / num_runs,
            "avg_head_1_usage": sum(head_1_usage) / num_runs,
            "head_1_usage_ratio": sum(head_1_usage) / sum(tokens_generated) if sum(tokens_generated) > 0 else 0
        }
        return results

    def interactive_generate(self):
        settings = {
            "max_new_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.95,
            "use_head_1_at_switch": True
        }

        while True:
            prompt = input("\n> Enter prompt: ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("Commands: quit, help, settings, set <param> <value>")
                continue
            elif prompt.lower() == 'settings':
                for k, v in settings.items():
                    print(f"  {k}: {v}")
                continue
            elif prompt.startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param in settings:
                        try:
                            if param == "use_head_1_at_switch":
                                settings[param] = value.lower() == 'true'
                            elif param == "max_new_tokens":
                                settings[param] = int(value)
                            else:
                                settings[param] = float(value)
                        except ValueError:
                            pass
                continue

            if prompt:
                output = self.generate(prompt, verbose=False, **settings)
                print(output.text)


def create_inference_engine(
        model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tpv_weights_path: Optional[str] = None,
        config: Optional[Config] = None
) -> DualHeadInference:
    if config is None:
        config = get_config()
    if model_path:
        config.model.slm_model_path = model_path
    model, tokenizer = create_dual_head_model(config)
    engine = DualHeadInference(model, tokenizer, config)
    if checkpoint_path:
        engine.load_checkpoint(checkpoint_path)
    if tpv_weights_path:
        engine.load_tpv_model(tpv_weights_path)
    return engine


def run_inference_examples(engine: DualHeadInference):
    examples = [
        "Let's solve this step by step.\n\n",
        "Problem: Find the value of x when 2x + 5 = 13\n\n",
        "To understand this concept, we need to consider:\n\n",
        "The solution can be found as follows:\n\n"
    ]
    for prompt in examples:
        output = engine.generate(prompt, max_new_tokens=50, temperature=0.6, use_head_1_at_switch=True, verbose=False)
