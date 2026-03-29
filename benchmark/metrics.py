"""
Timing and metrics collection for benchmarking.

Collects per-step metrics and produces JSON results + summary statistics.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    loss: float
    wall_time: float          # seconds since training start
    step_time_ms: float       # duration of this step in ms
    lr: float
    grad_norm: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete results for a single benchmark run."""
    model_name: str
    optimizer_name: str
    backend: str              # "olsvered" or "torch.linalg.inv" or "adam"
    total_steps: int
    total_time_s: float
    hyperparams: Dict
    step_metrics: List[StepMetrics] = field(default_factory=list)
    optimizer_timing: Dict = field(default_factory=dict)
    model_params: int = 0
    model_linear_layers: int = 0
    device: str = "cpu"
    seed: int = 42

    def add_step(self, step: int, loss: float, wall_time: float,
                 step_time_ms: float, lr: float, grad_norm: float = None):
        self.step_metrics.append(StepMetrics(
            step=step, loss=loss, wall_time=wall_time,
            step_time_ms=step_time_ms, lr=lr, grad_norm=grad_norm,
        ))

    def summary(self) -> Dict:
        """Generate summary statistics."""
        losses = [m.loss for m in self.step_metrics]
        step_times = [m.step_time_ms for m in self.step_metrics]
        return {
            "model": self.model_name,
            "optimizer": self.optimizer_name,
            "backend": self.backend,
            "total_steps": self.total_steps,
            "total_time_s": round(self.total_time_s, 2),
            "final_loss": round(losses[-1], 4) if losses else None,
            "min_loss": round(min(losses), 4) if losses else None,
            "mean_step_ms": round(np.mean(step_times), 2) if step_times else None,
            "p50_step_ms": round(np.percentile(step_times, 50), 2) if step_times else None,
            "p99_step_ms": round(np.percentile(step_times, 99), 2) if step_times else None,
            "model_params": self.model_params,
            "linear_layers": self.model_linear_layers,
            "device": self.device,
            "seed": self.seed,
            "hyperparams": self.hyperparams,
            "optimizer_timing": self.optimizer_timing,
        }

    def save(self, path: str):
        """Save full results to JSON."""
        data = {
            "summary": self.summary(),
            "steps": [asdict(m) for m in self.step_metrics],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
