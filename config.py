import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    slm_model_path: str = "path/to/DeepSeek-R1-Distill-Qwen-1.5B"
    llm_model_path: str = "path/to/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    special_vocab: List[str] = field(default_factory=lambda: [
        "The", "Thus", "Therefore", "So", "Then", "Let", "Wait", "Alternatively", "Now", "I", "First", "Option", "**", "-", "\[", "\\"
    ])
    hidden_size: Optional[int] = None
    head_1_output_dim: Optional[int] = None
    switch_delimiter: str = "\n\n"
    torch_dtype: torch.dtype = torch.float32
    device_map: str = "auto"

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    distill_temperature: float = 4.0
    distill_alpha: float = 0.7
    kl_loss_weight: float = 1.0
    warmup_steps: int = 50
    save_steps: int = 20
    eval_steps: int = 200000
    logging_steps: int = 200
    output_dir: str = "./checkpoints"
    save_total_limit: int = 50
    max_length: int = 2048
    num_proc: int = 4

@dataclass
class InferenceConfig:
    max_new_tokens: int = 2048
    chunk_size: int = 32
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = True
    use_head_1_at_switch: bool = True
    head_1_temperature: float = 0.6
    pad_token_id: Optional[int] = None

@dataclass
class DataConfig:
    dataset_name: str = "path/to/dataset"
    dataset_split: str = "train"
    num_samples: Optional[int] = None
    prompt_template: str = "### question:\n{question}\n\n### Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    val_split_ratio: float = 0.1
    val_max_samples: int = 100

@dataclass
class SystemConfig:
    device: str = "auto"
    num_gpus: int = 8
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    seed: int = 3
    deterministic: bool = True
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    experiment_name: str = "dual_head_distillation"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        self.model.head_1_output_dim = len(self.model.special_vocab)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        return cls(**config_dict)

    def to_dict(self) -> dict:
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__
        }

default_config = Config()

def get_config() -> Config:
    return default_config

def update_config(**kwargs) -> Config:
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            for attr_name in ['model', 'training', 'inference', 'data', 'system']:
                attr = getattr(config, attr_name)
                if hasattr(attr, key):
                    setattr(attr, key, value)
                    break
    return config
