from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from momo.filters import (
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import time_to_eps
from momo.noise import (
    GaussianNoise,
    MixedFARIMAStableNoise,
    PinkFARIMANoise,
    StableNoise,
)
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic


def build_filter(spec: dict):
    kind = spec["kind"]
    if kind == "identity":
        return IdentityFilter()
    if kind == "ma":
        return MovingAverageFilter(window=spec["window"])
    if kind == "kalman":
        return KalmanLocalLevelFilter(process_var=spec["process_var"], obs_var=spec["obs_var"])
    if kind == "wavelet":
        return WaveletThresholdFilter(wavelet=spec["wavelet"], mode=spec["mode"], threshold=spec["threshold"])
    if kind == "median":
        return MedianFilter(window=spec["window"])
    raise ValueError(kind)


def build_noise(spec: dict):
    kind = spec["kind"]
    if kind == "gaussian":
        return GaussianNoise(sigma=spec["sigma"])
    if kind == "pink":
        return PinkFARIMANoise(d=spec["d"], sigma=spec["sigma"])
    if kind == "stable":
        return StableNoise(alpha=spec["alpha"], sigma=spec["sigma"])
    if kind == "mixed":
        return MixedFARIMAStableNoise(d=spec["d"], alpha=spec["alpha"], sigma=spec["sigma"])
    raise ValueError(kind)


def build_task(name: str, spec: dict, seed: int):
    if name == "quadratic":
        return make_quadratic(dim=spec["dim"], condition_number=spec["condition_number"], seed=seed)
    if name == "logistic":
        return make_logistic(
            n=spec["n"], dim=spec["dim"], n_test=spec["n_test"],
            noise_scale=spec["noise_scale"], seed=seed,
        )
    raise ValueError(name)


def _holdout_metric(task_name: str, task, x_final: np.ndarray) -> float:
    if task_name == "logistic":
        from momo.tasks import _sigmoid
        logits = task.Z_test @ x_final
        preds = (_sigmoid(logits) >= 0.5).astype(int)
        return float(np.mean(preds == task.y_test))
    if task_name == "quadratic":
        return float(np.linalg.norm(x_final - task.optimum()))
    return float("nan")


def run_one(task_name, task_spec, filt_name, filt_spec, noise_name, noise_spec,
            opt_name, opt_spec, mode, seed, steps, buffer_size, batch_size,
            epsilon, runs_dir):
    task = build_task(task_name, task_spec, seed=seed)
    filt = build_filter(filt_spec)
    noise = build_noise(noise_spec)
    lr = task_spec.get("lr", opt_spec["lr"])
    t0 = time.perf_counter()
    result = run_optimization(
        task=task, optimizer=opt_name, noise=noise, filt=filt,
        steps=steps, lr=lr, batch_size=batch_size, seed=seed,
        buffer_size=buffer_size,
        weight_decay=opt_spec.get("weight_decay", 0.0),
        preprocess_mode=mode,
    )
    elapsed = time.perf_counter() - t0
    g_curve = result.grad_norm_sq_history
    t_eps = time_to_eps(g_curve, epsilon)
    holdout = _holdout_metric(task_name, task, result.x_final)
    cell_dir = runs_dir / mode / task_name / f"{filt_name}_{noise_name}_{opt_name}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cell_dir / f"seed{seed:02d}.npz",
        grad_norm_sq=g_curve.astype(np.float32),
        loss=result.loss_history.astype(np.float32),
        x_final=result.x_final.astype(np.float32),
    )
    return dict(
        mode=mode, task=task_name, filter=filt_name, noise=noise_name,
        optimizer=opt_name, seed=seed, steps=steps, lr=lr,
        t_eps=(-1 if t_eps is None else t_eps),
        final_grad_norm_sq=float(g_curve[-1]),
        final_loss=float(result.loss_history[-1]),
        holdout_metric=holdout, elapsed_s=elapsed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/synthetic.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/mode_compare"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/mode_compare_summary.csv"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    for mode in ["buffer", "data"]:
        for task_name, task_spec in cfg["tasks"].items():
            for filt_name, filt_spec in cfg["filters"].items():
                for noise_name, noise_spec in cfg["noises"].items():
                    for opt_name, opt_spec in cfg["optimizers"].items():
                        for seed in range(cfg["run"]["n_seeds"]):
                            cells.append((task_name, task_spec, filt_name, filt_spec,
                                          noise_name, noise_spec, opt_name, opt_spec,
                                          mode, seed))
    print(f"Running {len(cells)} cells (both modes) with n_jobs={args.n_jobs}")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(
            tn, ts, fn, fs, nn, ns, on, os_, mode, seed,
            cfg["run"]["steps"], cfg["run"]["buffer_size"],
            cfg["run"]["batch_size"], cfg["run"]["epsilon"], args.runs_dir,
        )
        for (tn, ts, fn, fs, nn, ns, on, os_, mode, seed) in cells
    )
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(args.summary_csv, index=False)
    print(f"Done in {elapsed:.1f}s. Wrote {args.summary_csv}")
    df["converged"] = df["t_eps"] != -1
    print("\nConvergence rate by mode/task/filter/noise:")
    print(df.groupby(["mode", "task", "filter", "noise"])["converged"].mean().unstack(["noise"]).round(2))


if __name__ == "__main__":
    main()
