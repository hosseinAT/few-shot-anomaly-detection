"""Few-shot normal-only fitting, independent calibration and held-out testing."""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import torch
from .data import SyntheticSurface, ManifestSurface, audit_manifest, stack_dataset
from .metrics import bootstrap_auroc, confusion, ranking, wilson_interval
from .models import Autoencoder, PatchMemory, conformal_threshold, image_scores
from .repro import environment, seed_all, write_json


@dataclass
class Config:
    seeds: list[int] = field(default_factory=lambda: [7, 17, 27])
    shots: list[int] = field(default_factory=lambda: [5, 15])
    budgets: list[int] = field(default_factory=lambda: [64, 256])
    train_pool: int = 40
    calibration: int = 32
    test: int = 80
    epochs: int = 25
    size: int = 64
    alpha: float = .05
    pixel_quantile: float = .995


@torch.inference_mode()
def score_maps(model, x):
    return torch.cat([model.score(batch) for batch in x.split(8)])


def evaluate(model, calibration, test, mask, alpha, pixel_quantile, seed):
    start = time.perf_counter()
    cal_maps, test_maps = score_maps(model, calibration), score_maps(model, test)
    calibration_seconds = time.perf_counter() - start
    cs, scores = image_scores(cal_maps), image_scores(test_maps)
    image_threshold = conformal_threshold(cs, alpha)
    pixel_threshold = float(torch.quantile(cal_maps.flatten(), pixel_quantile))
    y = mask.flatten(1).any(1).numpy().astype(int)
    s = scores.numpy()
    predicted = s > image_threshold
    normal, positive = y == 0, y == 1
    normal_fp = int(predicted[normal].sum())
    detected = int(predicted[positive].sum())
    metrics = {"image": {**confusion(predicted, y), **ranking(y, s)},
               "pixel": {**confusion(test_maps.numpy() > pixel_threshold, mask.numpy()), **ranking(mask.numpy(), test_maps.numpy())},
               "image_auroc_bootstrap_95": bootstrap_auroc(y, s, seed),
               "normal_false_alarm_rate": float(predicted[normal].mean()) if normal.any() else None,
               "normal_false_alarm_wilson_95": wilson_interval(normal_fp, int(normal.sum())),
               "defect_recall": float(predicted[positive].mean()) if positive.any() else None,
               "defect_recall_wilson_95": wilson_interval(detected, int(positive.sum())),
               "image_threshold": image_threshold, "image_threshold_finite": math.isfinite(image_threshold),
               "pixel_threshold": pixel_threshold, "calibration_and_test_seconds": calibration_seconds,
               "n_calibration": len(calibration), "normal_test": int(normal.sum()), "defect_test": int(positive.sum())}
    return metrics, test_maps, s


def figure(inputs, mask, maps, scores, threshold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    normal = ~mask.reshape(len(mask), -1).astype(bool).any(1)
    # Highest-scoring normal + lowest-scoring defect + two highest-score images.
    idx = []
    if normal.any():
        idx.append(int(np.flatnonzero(normal)[scores[normal].argmax()]))
    if (~normal).any():
        idx.append(int(np.flatnonzero(~normal)[scores[~normal].argmin()]))
    idx = list(dict.fromkeys(idx + np.argsort(scores)[::-1].tolist()))[:4]
    fig, axes = plt.subplots(len(idx), 3, figsize=(8, 2.3 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        for col, data in enumerate([inputs[i, 0], mask[i, 0], maps[i, 0]]):
            axes[row, col].imshow(data, cmap="magma" if col == 2 else "gray")
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{'normal' if normal[i] else 'defect'} / score {scores[i]:.3f}\n{'alarm' if scores[i] > threshold else 'no alarm'}", fontsize=9)
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Anomaly map\nper-image color scale", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(c, out, root=None):
    if not c.shots or not c.seeds or not c.budgets or min(c.shots) < 1 or min(c.budgets) < 1 or c.epochs < 1:
        raise ValueError("Invalid configuration")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit = audit_manifest(root) if root else None
    if root:
        tr, cal, te = [ManifestSurface(root, split, c.size, audit) for split in ("train", "cal", "test")]
    else:
        tr, cal, te = SyntheticSurface(c.train_pool, 51, c.size, True), SyntheticSurface(c.calibration, 52, c.size, True), SyntheticSurface(c.test, 53, c.size)
    train_x, train_y = stack_dataset(tr)
    cx, cy = stack_dataset(cal)
    tx, ty = stack_dataset(te)
    if train_y.any() or cy.any():
        raise ValueError("Fit/calibration splits must contain normal images only")
    if len(train_x) < max(c.shots):
        raise ValueError("Few-shot request exceeds available normal fit images")
    write_json(out / "config.json", asdict(c))
    write_json(out / "data_audit.json", audit or {"source": "procedural surfaces", "split_seeds": {"train": 51, "cal": 52, "test": 53}, "counts": {"normal_fit_pool": len(tr), "normal_calibration": len(cal), "test": len(te)}})
    rows = []
    for seed in c.seeds:
        permutation = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed))
        for shots in c.shots:
            seed_all(seed)
            selected = permutation[:shots]
            fit = train_x[selected]
            autoencoder = Autoencoder()
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=.003)
            losses = []
            for _ in range(c.epochs):
                total = 0.
                autoencoder.train()
                for batch in fit.split(8):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (autoencoder(batch) - batch).abs().mean()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                losses.append(total / shots)
            autoencoder.eval()
            methods = [("autoencoder", None, autoencoder)]
            for budget in c.budgets:
                methods += [("handcrafted_memory", budget, PatchMemory(budget).fit(fit)),
                            ("encoder_memory", budget, PatchMemory(budget, autoencoder).fit(fit))]
            for name, budget, model in methods:
                metrics, maps, scores = evaluate(model, cx, tx, ty, c.alpha, c.pixel_quantile, seed)
                tag = f"{name}_shots{shots}_k{budget or 0}_seed{seed}"
                record = {"method": name, "seed": seed, "shots": shots, "budget": budget,
                          "fit_sample_ids": [tr.sample_id(int(i)) for i in selected], "metrics": metrics}
                if isinstance(model, PatchMemory):
                    record.update({"bank_shape": list(model.bank.shape), "bank_and_normalization_bytes": model.storage_bytes(),
                                   "coverage_radius": model.coverage_radius, "fit_patch_count": model.fit_patch_count,
                                   "feature_source": "normal-only trained encoder" if model.encoder else "fixed intensity/gradient/texture descriptors"})
                    checkpoint = {"bank": model.bank, "mean": model.mean, "std": model.std,
                                  "encoder_weights": autoencoder.state_dict() if model.encoder else None}
                else:
                    record["loss"] = losses
                    record["parameters"] = sum(p.numel() for p in model.parameters())
                    checkpoint = {"autoencoder_weights": autoencoder.state_dict()}
                checkpoint.update({"method": name, "image_threshold": metrics["image_threshold"], "pixel_threshold": metrics["pixel_threshold"], "size": c.size})
                torch.save(checkpoint, out / (tag + ".pt"))
                rows.append(record)
                with (out / (tag + ".csv")).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_id", "kind", "score", "label", "alarm"])
                    for i, score in enumerate(scores):
                        writer.writerow([te.sample_id(i), te.kind(i), float(score), int(ty[i].any()), int(score > metrics["image_threshold"])])
                if seed == c.seeds[0] and shots == max(c.shots) and (budget == max(c.budgets) or name == "autoencoder"):
                    figure(tx.numpy(), ty.numpy(), maps.numpy(), scores, metrics["image_threshold"], out / (tag + ".png"))
                print(f"seed={seed} shots={shots} {name} k={budget} auroc={metrics['image']['auroc']} recall={metrics['defect_recall']}", flush=True)
    result = {"data": "private manifest" if root else "procedural synthetic only", "config": asdict(c), "runs": rows,
              "environment": environment(Path(__file__).parent),
              "budget_definition": "shots counts normal fit images; independent calibration images are additional.",
              "feature_claim": "No pretrained ImageNet backbone; this is a patch-memory study, not a PatchCore reproduction."}
    write_json(out / "metrics.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--root")
    args = p.parse_args()
    run(Config(**json.loads(Path(args.config).read_text())), args.out, args.root)


if __name__ == "__main__":
    main()
