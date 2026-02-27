import argparse
import sys
import os
from pathlib import Path
import torch
from transformers import set_seed
import json
from typing import Optional, List
import time

sys.path.append('.')

from config import Config, get_config
from model import create_dual_head_model
from inference import create_inference_engine, run_inference_examples


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with Dual-Head Distillation Model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--head_1_temperature", type=float, default=0.6)
    parser.add_argument("--no_head_1", action="store_true")
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_sample", dest="do_sample", action="store_false")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_prompts(prompt_file: str) -> List[str]:
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def save_outputs(outputs: List, output_file: str):
    with open(output_file, 'w') as f:
        for i, output in enumerate(outputs):
            f.write(f"{'='*50}\n")
            f.write(f"Generation {i+1}\n")
            f.write(f"{'='*50}\n")

            if hasattr(output, 'metadata') and 'prompt' in output.metadata:
                f.write(f"Prompt: {output.metadata['prompt']}\n")
                f.write(f"{'-'*30}\n")

            f.write(f"{output.text}\n")

            if hasattr(output, 'num_tokens'):
                f.write(f"\n{'-'*30}\n")
                f.write(f"Tokens: {output.num_tokens}, Time: {output.generation_time:.2f}s, ")
                f.write(f"Tok/s: {output.tokens_per_second:.1f}, Head_1: {len(output.head_1_tokens)}\n")

            f.write("\n\n")


def run_single_generation(engine, args):
    prompt = args.prompt or input("Enter prompt: ")

    output = engine.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_head_1_at_switch=not args.no_head_1,
        head_1_temperature=args.head_1_temperature,
        verbose=False
    )

    if args.output_file:
        save_outputs([output], args.output_file)

    return output


def run_batch_generation(engine, args):
    if args.prompt_file:
        prompts = load_prompts(args.prompt_file)
    else:
        prompts = []
        while True:
            prompt = input(f"Prompt {len(prompts)+1}: ").strip()
            if not prompt:
                break
            prompts.append(prompt)

    if not prompts:
        return

    outputs = engine.generate_batch(
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_head_1_at_switch=not args.no_head_1,
        head_1_temperature=args.head_1_temperature,
        verbose=False
    )

    if args.output_file:
        save_outputs(outputs, args.output_file)

    return outputs


def main():
    args = parse_arguments()
    set_seed(args.seed)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = get_config()

    if args.model_path:
        config.model.slm_model_path = args.model_path

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config.model.device_map = device

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        return 1

    try:
        engine = create_inference_engine(
            model_path=config.model.slm_model_path,
            checkpoint_path=str(checkpoint_path),
            config=config
        )

        if args.interactive:
            engine.interactive_generate()

        elif args.benchmark:
            if not args.prompt:
                args.prompt = "Let's solve this problem step by step.\n\n"

            results = engine.benchmark(
                prompt=args.prompt,
                num_runs=5,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                use_head_1_at_switch=not args.no_head_1,
                head_1_temperature=args.head_1_temperature
            )

            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)

        elif args.examples:
            run_inference_examples(engine)

        elif args.prompt_file or (args.prompt and '\n' in args.prompt):
            outputs = run_batch_generation(engine, args)

        else:
            output = run_single_generation(engine, args)

    except KeyboardInterrupt:
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
