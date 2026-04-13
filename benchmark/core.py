"""
Shared BenchmarkRunner abstraction for all OlsSM benchmark scripts.

Provides a single, parameterised training loop that all benchmark scripts
can call instead of duplicating forward/backward/timing logic.  The existing
scripts (gpu_benchmark.py, training_benchmark.py, run.py) remain working —
this module is an additive abstraction they can gradually migrate to.

Usage
-----
::

    from benchmark.core import BenchmarkConfig, BenchmarkRunner, BenchmarkResult

    cfg = BenchmarkConfig(
        name="OlsSMKFAC",
        model=model,
        optimizer=opt,
        dataloader=loader,
        steps=500,
    )
    runner = BenchmarkRunner(cfg)
    result = runner.run()
    result.print_summary()
    result.save("results/my_run.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """All parameters needed to run a single benchmark trial.

    Parameters
    ----------
    name : str
        Human-readable optimizer name (used in plots and summaries).
    model : nn.Module
        The model to benchmark.
    optimizer : torch.optim.Optimizer
        Optimizer wrapping the model's parameters.
    dataloader : Iterable
        Yields ``(batch_x, batch_y)`` tuples.
    steps : int
        Total gradient steps to run.
    loss_fn : Callable, optional
        ``loss_fn(logits, targets) → scalar Tensor``.
        Defaults to ``nn.CrossEntropyLoss()``.
    eval_fn : Callable, optional
        ``eval_fn(model, dataloader) → (val_loss, val_metric)`` called every
        ``eval_every`` steps.  Returns ``(float, float)``.
        If None, validation is skipped.
    eval_every : int
        How often (in steps) to call ``eval_fn``.  Default: 50.
    scheduler : optional
        LR scheduler with a ``.step()`` method called after each optimizer step.
    device : torch.device, optional
        Device to move batches to.  Inferred from model if not provided.
    log_every : int
        Print progress every N steps.  0 = silent.  Default: 50.
    extra : dict
        Arbitrary metadata stored verbatim in the result JSON.
    """
    name: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataloader: Iterable
    steps: int
    loss_fn: Optional[Callable] = None
    eval_fn: Optional[Callable] = None
    eval_every: int = 50
    scheduler: Optional[Any] = None
    device: Optional[torch.device] = None
    log_every: int = 50
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Collected metrics from a single benchmark run.

    All ``curve_*`` lists are parallel — index i corresponds to
    the i-th evaluation checkpoint.
    """
    name: str
    steps: int = 0
    samples: int = 0
    wall_s: float = 0.0

    # Per-step timing (seconds)
    fwdbwd_times: List[float] = field(default_factory=list)
    opt_times: List[float] = field(default_factory=list)

    # Evaluation curves (recorded every eval_every steps)
    curve_steps: List[int] = field(default_factory=list)
    curve_times: List[float] = field(default_factory=list)
    curve_val_loss: List[float] = field(default_factory=list)
    curve_val_metric: List[float] = field(default_factory=list)   # acc or ppl

    # Peak GPU memory
    peak_mem_gb: float = 0.0

    # Metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def avg_fwdbwd_ms(self) -> float:
        if not self.fwdbwd_times:
            return 0.0
        return 1000.0 * sum(self.fwdbwd_times) / len(self.fwdbwd_times)

    @property
    def avg_opt_ms(self) -> float:
        if not self.opt_times:
            return 0.0
        return 1000.0 * sum(self.opt_times) / len(self.opt_times)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "steps":            self.steps,
            "samples":          self.samples,
            "wall_s":           self.wall_s,
            "avg_fwdbwd_ms":    self.avg_fwdbwd_ms,
            "avg_opt_ms":       self.avg_opt_ms,
            "peak_mem_gb":      self.peak_mem_gb,
            "curve_steps":      self.curve_steps,
            "curve_times":      self.curve_times,
            "curve_val_loss":   self.curve_val_loss,
            "curve_val_metric": self.curve_val_metric,
            **self.extra,
        }

    def save(self, path: str) -> None:
        """Write result to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print a one-line summary to stdout."""
        final_metric = self.curve_val_metric[-1] if self.curve_val_metric else float("nan")
        final_loss   = self.curve_val_loss[-1]   if self.curve_val_loss   else float("nan")
        print(
            f"  [{self.name:<16}]  steps={self.steps:,}  "
            f"wall={self.wall_s:.1f}s  "
            f"fwd+bwd={self.avg_fwdbwd_ms:.1f}ms  "
            f"opt={self.avg_opt_ms:.1f}ms  "
            f"mem={self.peak_mem_gb:.2f}GB  "
            f"val_loss={final_loss:.4f}  "
            f"val_metric={final_metric:.4f}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Executes a single benchmark trial described by a :class:`BenchmarkConfig`.

    The run loop:
    1. Iterates ``config.steps`` gradient steps over ``config.dataloader``
       (cycling back if the dataloader is exhausted).
    2. Times forward+backward and optimizer step separately.
    3. Calls ``eval_fn`` every ``eval_every`` steps and records curves.
    4. Returns a :class:`BenchmarkResult`.

    Example
    -------
    ::

        runner = BenchmarkRunner(cfg)
        result = runner.run()
        result.print_summary()
        result.save("results/trial.json")
    """

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self._loss_fn = config.loss_fn or nn.CrossEntropyLoss()
        self._device  = (
            config.device
            or next(config.model.parameters()).device
        )

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark and return results."""
        cfg = self.cfg
        result = BenchmarkResult(name=cfg.name, extra=dict(cfg.extra))

        data_iter = self._infinite(cfg.dataloader)
        t_start   = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step in range(1, cfg.steps + 1):
            batch_x, batch_y = next(data_iter)
            batch_x = batch_x.to(self._device)
            batch_y = batch_y.to(self._device)

            # ── Forward + backward ────────────────────────────────────────
            t0 = time.perf_counter()
            cfg.model.zero_grad()
            logits = cfg.model(batch_x)
            loss   = self._loss_fn(logits, batch_y)
            loss.backward()
            result.fwdbwd_times.append(time.perf_counter() - t0)

            # ── Optimizer step ────────────────────────────────────────────
            t1 = time.perf_counter()
            cfg.optimizer.step()
            if cfg.scheduler is not None:
                cfg.scheduler.step()
            result.opt_times.append(time.perf_counter() - t1)

            result.steps   += 1
            result.samples += batch_x.shape[0]

            # ── Evaluation ────────────────────────────────────────────────
            if cfg.eval_fn is not None and step % cfg.eval_every == 0:
                val_loss, val_metric = cfg.eval_fn(cfg.model, cfg.dataloader)
                wall = time.perf_counter() - t_start
                result.curve_steps.append(step)
                result.curve_times.append(wall)
                result.curve_val_loss.append(val_loss)
                result.curve_val_metric.append(val_metric)

                if cfg.log_every > 0 and step % cfg.log_every == 0:
                    print(
                        f"  step {step:>5}/{cfg.steps}  "
                        f"loss={val_loss:.4f}  metric={val_metric:.4f}  "
                        f"opt={result.avg_opt_ms:.1f}ms"
                    )

            elif cfg.log_every > 0 and step % cfg.log_every == 0:
                wall = time.perf_counter() - t_start
                print(
                    f"  step {step:>5}/{cfg.steps}  "
                    f"train_loss={loss.item():.4f}  "
                    f"opt={result.avg_opt_ms:.1f}ms  "
                    f"wall={wall:.1f}s"
                )

        result.wall_s = time.perf_counter() - t_start

        if torch.cuda.is_available():
            result.peak_mem_gb = (
                torch.cuda.max_memory_allocated() / 1024 ** 3
            )

        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _infinite(dataloader: Iterable) -> Iterable:
        """Cycle a dataloader indefinitely."""
        while True:
            yield from dataloader
