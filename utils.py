import torch
import logging
import sys
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import psutil
import GPUtil
import subprocess


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )

    logger = logging.getLogger("dual_head")
    logger.setLevel(getattr(logging, log_level.upper()))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def get_device_info() -> Dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "devices": []
    }

    if info["cuda_available"]:
        for i in range(info["cuda_count"]):
            device_info = {
                "type": "cuda",
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / 1e9,
                "memory_allocated": torch.cuda.memory_allocated(i) / 1e9,
                "memory_reserved": torch.cuda.memory_reserved(i) / 1e9
            }
            info["devices"].append(device_info)

    cpu_info = {
        "type": "cpu",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / 1e9,
        "memory_available": psutil.virtual_memory().available / 1e9
    }
    info["devices"].append(cpu_info)

    return info


def get_memory_usage() -> Dict:
    stats = {}

    cpu_mem = psutil.virtual_memory()
    stats["cpu"] = {
        "used_gb": (cpu_mem.total - cpu_mem.available) / 1e9,
        "total_gb": cpu_mem.total / 1e9,
        "percent": cpu_mem.percent
    }

    if torch.cuda.is_available():
        stats["gpu"] = []
        for i in range(torch.cuda.device_count()):
            gpu_stats = {
                "device": i,
                "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                "total_gb": torch.cuda.get_device_properties(i).total_memory / 1e9
            }
            stats["gpu"].append(gpu_stats)

    return stats


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def ensure_directory(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_mb": total * 4 / 1e6,
        "trainable_mb": trainable * 4 / 1e6
    }


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_training_time(
    steps_done: int,
    total_steps: int,
    time_elapsed: float
) -> Tuple[float, str]:
    if steps_done == 0:
        return 0, "N/A"

    time_per_step = time_elapsed / steps_done
    steps_remaining = total_steps - steps_done
    time_remaining = time_per_step * steps_remaining

    return time_remaining, format_time(time_remaining)


class MetricsTracker:

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key] = self.metrics[key][-self.window_size:]

    def get_average(self, key: str) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_last(self, key: str) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]

    def get_summary(self) -> Dict:
        summary = {}
        for key in self.metrics:
            if self.metrics[key]:
                summary[f"{key}_avg"] = self.get_average(key)
                summary[f"{key}_last"] = self.get_last(key)
        return summary

    def reset(self):
        self.metrics = {}


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_prompt(prompt: str, max_display_length: int = 50) -> str:
    display_prompt = prompt.replace('\n', '\\n')
    return truncate_text(display_prompt, max_display_length)


def count_tokens_approx(text: str) -> int:
    return len(text) // 4


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        if best_checkpoint.exists():
            return best_checkpoint
        return None

    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    return checkpoints[-1]


def list_checkpoints(checkpoint_dir: Union[str, Path]) -> List[Dict]:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    for ckpt_path in checkpoint_dir.glob("*.pt"):
        try:
            stat = ckpt_path.stat()
            info = {
                "path": str(ckpt_path),
                "name": ckpt_path.name,
                "size_mb": stat.st_size / 1e6,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }

            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if isinstance(ckpt, dict):
                    info["epoch"] = ckpt.get("epoch", None)
                    info["step"] = ckpt.get("global_step", None)
                    if "metrics" in ckpt:
                        info["loss"] = ckpt["metrics"].get("avg_loss", None)
            except:
                pass

            checkpoints.append(info)
        except Exception:
            pass

    checkpoints.sort(key=lambda x: x["modified"], reverse=True)

    return checkpoints


class ExperimentTracker:

    def __init__(self, experiment_dir: Union[str, Path]):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.experiment_dir / "experiment.log"
        self.metrics_file = self.experiment_dir / "metrics.json"

        self.start_time = datetime.now()
        self.metrics = {
            "start_time": self.start_time.isoformat(),
            "updates": []
        }

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def update_metrics(self, **kwargs):
        update = {
            "timestamp": datetime.now().isoformat(),
            "elapsed": (datetime.now() - self.start_time).total_seconds(),
            **kwargs
        }

        self.metrics["updates"].append(update)
        save_json(self.metrics, self.metrics_file)

    def finish(self):
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["total_duration"] = (datetime.now() - self.start_time).total_seconds()
        save_json(self.metrics, self.metrics_file)


def get_git_revision() -> Optional[str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return revision[:7]
    except:
        return None


def get_system_info() -> Dict:
    import platform

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "git_revision": get_git_revision(),
        "hostname": platform.node()
    }

    return info
