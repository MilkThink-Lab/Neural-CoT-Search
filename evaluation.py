import re
import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from datetime import datetime
from datasets import load_dataset
import sys

sys.path.append('.')

from config import Config, get_config
from inference import create_inference_engine


@dataclass
class EvaluationResult:
    problem_id: str
    question: str
    reference_answer: Union[str, float]
    predicted_answer: Union[str, float]
    is_correct: bool
    generated_text: str
    generation_time: float
    problem_type: str
    metadata: Dict = field(default_factory=dict)


class ProblemFormatter:
    @staticmethod
    def format_multiple_choice(question: str, choices: List) -> str:
        formatted_text = f"### question:\n{question}\n\n### choices:\n"
        for label, text in zip(['A', 'B', 'C', 'D'], choices):
            formatted_text += f"{label}. {text}\n"
        formatted_text += "\n### Please reason step by step, and put your final answer within \\boxed{}.\n<think>"
        return formatted_text

    @staticmethod
    def format_math(question: str) -> str:
        return f"### question:\n{question}\n\n### Please reason step by step, and put your final answer within \\boxed{{}}.\n<think>"


class AnswerExtractor:
    MC_PATTERNS = [
        r"\\boxed\{([A-D])\}",
        r"\bthe answer is\s+([A-D])\b",
        r"\bfinal answer[: ]\s*([A-D])\b",
    ]

    MATH_PATTERNS = [
        r"\\boxed\{([^}]*)\}",
        r"\bthe answer is\s+([-+]?\d*\.?\d+)\b",
        r"\bfinal answer[: ]\s*([-+]?\d*\.?\d+)\b"
    ]

    @classmethod
    def extract_multiple_choice(cls, text: str) -> Optional[str]:
        for pattern in cls.MC_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    answer = match.group(1)
                    if answer and answer.upper() in ['A', 'B', 'C', 'D']:
                        return answer.upper()
        all_letters = re.findall(r"\b([A-D])\b", text, re.IGNORECASE)
        if all_letters:
            return all_letters[-1].upper()
        return None

    @classmethod
    def extract_math_answer(cls, text: str) -> Optional[float]:
        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed_matches:
            last_boxed = boxed_matches[-1]
            numbers = re.findall(r"[-+]?\d*\.?\d+", last_boxed)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass

        for pattern in cls.MATH_PATTERNS[1:]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            last_match = None
            for match in matches:
                last_match = match
            if last_match and last_match.groups():
                try:
                    return float(last_match.group(1))
                except (ValueError, IndexError):
                    pass

        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None


class DatasetEvaluator:
    def __init__(self, model_engine, config: Optional[Config] = None, verbose: bool = False):
        self.engine = model_engine
        self.config = config or get_config()
        self.verbose = verbose
        self.formatter = ProblemFormatter()
        self.extractor = AnswerExtractor()

    def evaluate_problem(
            self,
            problem: Dict,
            problem_type: str,
            max_new_tokens: int = 4096,
            max_reasoning_steps: int = 50,
            force_summary_tokens: int = 100
    ) -> EvaluationResult:
        if problem_type == "multiple_choice":
            if "choices" in problem:
                formatted_input = self.formatter.format_multiple_choice(
                    problem.get("question", problem.get("problem", "")),
                    problem["choices"]['text']
                )
            elif 'options' in problem:
                formatted_input = self.formatter.format_multiple_choice(
                    problem.get("question", problem.get("problem", "")),
                    problem["options"]
                )
            else:
                raise ValueError("Multiple choice problem missing 'choices' field")
            reference_answer = problem.get("answer", problem.get("answerKey", ""))
            if reference_answer in [0, 1, 2, 3]:
                reference_answer = ['A', 'B', 'C', 'D'][reference_answer]
        else:
            formatted_input = self.formatter.format_math(
                problem.get("question", problem.get("problem", problem.get("Problem", "")))
            )
            reference_answer = problem.get("answer", problem.get("solution", problem.get("Answer", "")))
            if type(reference_answer) == str:
                try:
                    reference_answer = float(reference_answer)
                except:
                    reference_answer = float(reference_answer.split("#### ")[-1].replace(",", ""))

        import time
        start_time = time.time()

        generation_output = self.engine.generate(
            prompt=formatted_input,
            max_new_tokens=max_new_tokens,
            max_paragraphs=max_reasoning_steps,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            use_head_1_at_switch=self.config.inference.use_head_1_at_switch,
            verbose=False
        )

        generated_text = generation_output.text
        generation_time = time.time() - start_time

        early_stopped = generation_output.metadata.get("early_stop_reason") == "max_paragraphs_reached"
        actual_paragraph_count = generation_output.metadata.get("final_paragraph_count", 0)

        if early_stopped:
            continuation_prompt = "\n\nYou have reached the maximum reasoning steps. Please summarize your reasoning and provide the final answer immediately:"
            continuation_input = generated_text + continuation_prompt

            continuation_output = self.engine.generate(
                prompt=continuation_input,
                max_new_tokens=force_summary_tokens,
                max_paragraphs=None,
                temperature=self.config.inference.temperature,
                top_p=self.config.inference.top_p,
                use_head_1_at_switch=self.config.inference.use_head_1_at_switch,
                verbose=False
            )

            generated_text = continuation_output.text
            generation_time = time.time() - start_time

            total_tokens = generation_output.num_tokens + continuation_output.num_tokens
            total_head_1_tokens = len(generation_output.head_1_tokens) + len(continuation_output.head_1_tokens)
            metadata = {
                "num_tokens": total_tokens,
                "head_1_usage": total_head_1_tokens,
                "truncated": True,
                "reasoning_steps": actual_paragraph_count,
                "early_stopped": True,
                "max_reasoning_steps_reached": True
            }
        else:
            metadata = {
                "num_tokens": generation_output.num_tokens,
                "head_1_usage": len(generation_output.head_1_tokens),
                "truncated": False,
                "reasoning_steps": actual_paragraph_count,
                "early_stopped": False,
                "max_reasoning_steps_reached": False
            }

        if problem_type == "multiple_choice":
            predicted_answer = self.extractor.extract_multiple_choice(generated_text)
            is_correct = predicted_answer == str(reference_answer).upper() if predicted_answer else False
        else:
            predicted_answer = self.extractor.extract_math_answer(generated_text)
            if predicted_answer is not None and reference_answer:
                try:
                    ref_value = float(reference_answer) if isinstance(reference_answer, (int, str)) else reference_answer
                    is_correct = abs(predicted_answer - ref_value) < 1e-6
                except (ValueError, TypeError):
                    is_correct = False
            else:
                is_correct = False

        return EvaluationResult(
            problem_id=problem.get("id", "unknown"),
            question=problem.get("question", problem.get("problem", "")),
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            generated_text=generated_text,
            generation_time=generation_time,
            problem_type=problem_type,
            metadata=metadata
        )

    def evaluate_dataset(
            self,
            dataset_name: str,
            name: str = None,
            split: str = "test",
            num_samples: Optional[int] = None,
            problem_type: str = "auto",
            max_new_tokens: int = 4096,
            max_reasoning_steps: int = 50,
            force_summary_tokens: int = 100,
            save_results: bool = True,
            output_dir: Optional[str] = None
    ) -> Dict:
        try:
            dataset = load_dataset(dataset_name, name=name, split=split)
        except Exception as e:
            return {}

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        if problem_type == "auto":
            sample = dataset[0]
            if "choices" in sample or "options" in sample:
                problem_type = "multiple_choice"
            else:
                problem_type = "math"

        results = []
        correct_count = 0
        truncated_count = 0

        progress_bar = tqdm(dataset, desc="Evaluating")
        for i, problem in enumerate(progress_bar):
            try:
                result = self.evaluate_problem(
                    problem, problem_type, max_new_tokens, max_reasoning_steps, force_summary_tokens
                )
                results.append(result)

                if result.is_correct:
                    correct_count += 1
                if result.metadata.get("truncated", False):
                    truncated_count += 1

                accuracy = correct_count / (i + 1) * 100
                progress_bar.set_postfix({
                    "correct": correct_count,
                    "accuracy": f"{accuracy:.1f}%",
                    "truncated": truncated_count
                })

            except Exception as e:
                continue

        total = len(results)
        accuracy = (correct_count / total * 100) if total > 0 else 0

        stats = {
            "dataset": dataset_name,
            "split": split,
            "problem_type": problem_type,
            "total_problems": total,
            "correct": correct_count,
            "incorrect": total - correct_count,
            "accuracy": accuracy,
            "truncated_count": truncated_count,
            "truncation_rate": (truncated_count / total * 100) if total > 0 else 0,
            "max_reasoning_steps": max_reasoning_steps,
            "avg_generation_time": np.mean([r.generation_time for r in results]) if results else 0,
            "avg_tokens": np.mean([r.metadata.get("num_tokens", 0) for r in results]) if results else 0,
            "avg_head_1_usage": np.mean([r.metadata.get("head_1_usage", 0) for r in results]) if results else 0,
            "avg_reasoning_steps": np.mean([r.metadata.get("reasoning_steps", 0) for r in results]) if results else 0
        }

        if save_results:
            self.save_results(results, stats, output_dir)

        return stats

    def save_results(self, results: List[EvaluationResult], stats: Dict, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = Path("./evaluation_results")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = stats["dataset"].replace("/", "_")

        stats_file = output_dir / f"{dataset_name}_{timestamp}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        results_file = output_dir / f"{dataset_name}_{timestamp}_results.json"
        results_data = []
        for r in results:
            results_data.append({
                "problem_id": r.problem_id,
                "question": r.question,
                "reference_answer": str(r.reference_answer),
                "predicted_answer": str(r.predicted_answer),
                "is_correct": r.is_correct,
                "generation_time": r.generation_time,
                "problem_type": r.problem_type,
                "metadata": r.metadata
            })
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        samples_file = output_dir / f"{dataset_name}_{timestamp}_samples.txt"
        with open(samples_file, 'w') as f:
            incorrect_samples = [r for r in results if not r.is_correct][:10]
            correct_samples = [r for r in results if r.is_correct][:10]

            f.write("INCORRECT SAMPLES\n" + "=" * 60 + "\n\n")
            for i, r in enumerate(incorrect_samples, 1):
                f.write(f"Sample {i}:\nQuestion: {r.question}\nReference: {r.reference_answer}\nPredicted: {r.predicted_answer}\n\nGenerated:\n{r.generated_text}\n\n" + "-" * 60 + "\n\n")

            f.write("\nCORRECT SAMPLES\n" + "=" * 60 + "\n\n")
            for i, r in enumerate(correct_samples, 1):
                f.write(f"Sample {i}:\nQuestion: {r.question}\nReference: {r.reference_answer}\nPredicted: {r.predicted_answer}\n\nGenerated:\n{r.generated_text}\n\n" + "-" * 60 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Head Model on Datasets")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tpv_weights", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--problem_type", type=str, choices=["multiple_choice", "math", "auto"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_reasoning_steps", type=int, default=50)
    parser.add_argument("--force_summary_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--no_head_1", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = get_config()

    if args.model_path:
        config.model.slm_model_path = args.model_path
    config.inference.temperature = args.temperature
    config.inference.top_p = args.top_p
    config.inference.use_head_1_at_switch = not args.no_head_1

    engine = create_inference_engine(
        model_path=config.model.slm_model_path,
        checkpoint_path=args.checkpoint,
        tpv_weights_path=args.tpv_weights,
        config=config
    )

    evaluator = DatasetEvaluator(model_engine=engine, config=config, verbose=args.verbose)

    stats = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        name=args.name,
        split=args.split,
        num_samples=args.num_samples,
        problem_type=args.problem_type,
        max_new_tokens=args.max_new_tokens,
        max_reasoning_steps=args.max_reasoning_steps,
        force_summary_tokens=args.force_summary_tokens,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )

    return 0 if stats.get("accuracy", 0) > 0 else 1


if __name__ == "__main__":
    exit(main())
