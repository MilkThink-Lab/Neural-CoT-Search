import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import random

from config import Config, get_config


class PromptFormatter:
    def __init__(self, template: str):
        self.template = template

    def format(self, question: str) -> str:
        return self.template.format(question=question)


class DualHeadDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, config: Config, is_train: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.prompt_formatter = PromptFormatter(config.data.prompt_template)
        self.processed_data = self._process_data()

    def _process_data(self) -> List[Dict]:
        processed = []
        for item in self.data:
            if "question" in item:
                prompt = self.prompt_formatter.format(item["question"])
            elif "problem" in item:
                prompt = self.prompt_formatter.format(item["problem"])
            elif "text" in item:
                prompt = item["text"]
            else:
                continue

            input_ids = self.tokenizer.encode(
                prompt,
                max_length=self.config.training.max_length,
                truncation=True,
                add_special_tokens=True
            )

            processed_item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "prompt": prompt,
                "original": item
            }

            if "solution" in item:
                processed_item["solution"] = item["solution"]
            elif "answer" in item:
                processed_item["solution"] = item["answer"]

            processed.append(processed_item)
        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict:
        input_ids = [f["input_ids"] for f in features]
        max_length = max(len(ids) for ids in input_ids)

        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                         // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            mask = torch.cat([
                torch.ones(len(ids), dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "prompts": [f["prompt"] for f in features]
        }

        if "solution" in features[0]:
            batch["solutions"] = [f.get("solution", "") for f in features]

        return batch


def load_hf_dataset(config: Config) -> Tuple[HFDataset, Optional[HFDataset]]:
    try:
        dataset = load_dataset(config.data.dataset_name, split=config.data.dataset_split)

        if config.data.num_samples is not None:
            dataset = dataset.select(range(min(config.data.num_samples, len(dataset))))

        if config.data.val_split_ratio > 0:
            split_dataset = dataset.train_test_split(
                test_size=config.data.val_split_ratio,
                seed=config.system.seed
            )
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]

            if len(val_dataset) > config.data.val_max_samples:
                val_dataset = val_dataset.select(range(config.data.val_max_samples))

            return train_dataset, val_dataset
        else:
            return dataset, None

    except Exception as e:
        dummy_data = [
            {"question": f"Problem {i}: Solve for x in the equation {i}x + {i+1} = {i+10}"}
            for i in range(100)
        ]
        dataset = HFDataset.from_list(dummy_data)

        if config.data.val_split_ratio > 0:
            split_dataset = dataset.train_test_split(
                test_size=config.data.val_split_ratio,
                seed=config.system.seed
            )
            return split_dataset["train"], split_dataset["test"]
        else:
            return dataset, None


def create_dataloaders(config: Config, tokenizer: AutoTokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_hf, val_hf = load_hf_dataset(config)

    train_data = [item for item in train_hf]
    val_data = [item for item in val_hf] if val_hf else None

    train_dataset = DualHeadDataset(train_data, tokenizer, config, is_train=True)
    val_dataset = DualHeadDataset(val_data, tokenizer, config, is_train=False) if val_data else None

    collator = DataCollator(tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.system.dataloader_num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.system.dataloader_num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def prepare_batch_for_generation(
    batch: Dict,
    tokenizer: AutoTokenizer,
    device: Union[str, torch.device]
) -> List[torch.Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if isinstance(device, str):
        device = torch.device(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    sequences = []
    for i in range(input_ids.shape[0]):
        actual_length = attention_mask[i].sum().item()
        seq = input_ids[i, :actual_length].unsqueeze(0)
        sequences.append(seq)

    return sequences


def create_generation_prompt(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    return input_ids


def analyze_dataset(dataset: Union[HFDataset, DualHeadDataset]) -> Dict:
    stats = {
        "num_samples": len(dataset),
        "fields": [],
        "avg_length": 0,
        "max_length": 0,
        "min_length": float('inf')
    }

    if isinstance(dataset, HFDataset) and len(dataset) > 0:
        stats["fields"] = list(dataset[0].keys())
        sample_size = min(100, len(dataset))
        sample_indices = random.sample(range(len(dataset)), sample_size)
        lengths = []
        for idx in sample_indices:
            item = dataset[idx]
            text = item.get("question", item.get("problem", item.get("text", "")))
            lengths.append(len(text))
        if lengths:
            stats["avg_length"] = np.mean(lengths)
            stats["max_length"] = max(lengths)
            stats["min_length"] = min(lengths)

    elif isinstance(dataset, DualHeadDataset):
        if dataset.processed_data:
            lengths = [len(item["input_ids"]) for item in dataset.processed_data]
            stats["avg_length"] = np.mean(lengths)
            stats["max_length"] = max(lengths)
            stats["min_length"] = min(lengths)

    return stats


def sample_dataset(dataset: Union[HFDataset, DualHeadDataset], num_samples: int = 3) -> List[str]:
    samples = []
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for idx in indices:
        if isinstance(dataset, HFDataset):
            item = dataset[idx]
            text = str(item)
        elif isinstance(dataset, DualHeadDataset):
            item = dataset[idx]
            text = item.get("prompt", str(item))
        else:
            text = "Unknown format"
        samples.append(text)
    return samples
