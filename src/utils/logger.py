"""Unified logging: WandB + TensorBoard."""

from __future__ import annotations
from pathlib import Path
import time


class Logger:
    '''logs metrics to stdout, wandb, and tensorboard'''

    def __init__(
        self,
        project: str = "humanoid-rl",
        entity: str | None = None,
        run_name: str | None = None,
        log_dir: str = "logs",
        use_wandb: bool = True,
        use_tb: bool = True,
        config: dict | None = None,
    ):
        self._step = 0
        self._start_time = time.time()
        self._wandb_run = None
        self._tb_writer = None

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
            except Exception as e:
                print(f"[logger] wandb init failed: {e}, continuing without wandb")

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(log_dir) / (run_name or "default")
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(str(tb_dir))
            except Exception as e:
                print(f"[logger] tensorboard init failed: {e}, continuing without tb")

    def log(self, metrics: dict, step: int | None = None):
        '''log a dict of scalar metrics'''
        if step is not None:
            self._step = step
        else:
            self._step += 1

        # wandb
        if self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self._step)

        # tensorboard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, float(v), self._step)
            self._tb_writer.flush()

    def print_metrics(self, metrics: dict, step: int, total_steps: int):
        '''pretty print a metrics dict to stdout'''
        elapsed = time.time() - self._start_time
        fps = step / max(elapsed, 1e-6)
        parts = [f"step {step}/{total_steps}"]
        parts.append(f"fps={fps:.0f}")
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))

    def close(self):
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()
