#!/usr/bin/env python
"""Train tied SAE L1 sweep on cached activations (Phase 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from multilingual_mechinterp.sae import sweep_l1, train_tied_sae
from multilingual_mechinterp.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pythia70m_sae.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--single-alpha", type=float, default=None, help="Train one SAE instead of a sweep")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.exists() else {}
    data_cfg = cfg.get("data", {})
    sae_cfg = cfg.get("sae", {})
    results_cfg = cfg.get("results", {})
    model_cfg = cfg.get("model", {})

    dataset_dir = args.dataset_dir or Path(
        data_cfg.get("activations_dir", "data/processed/activations/layer_1")
    )
    device = args.device or model_cfg.get("device", "cpu")
    ratio = args.ratio if args.ratio is not None else float(sae_cfg.get("ratio", 2))
    n_epochs = args.n_epochs if args.n_epochs is not None else int(sae_cfg.get("n_epochs", 1))
    lr = float(sae_cfg.get("lr", 1e-3))
    batch_size = int(sae_cfg.get("batch_size", 512))

    if args.single_alpha is not None:
        alpha = float(args.single_alpha)
        result = train_tied_sae(
            dataset_dir,
            ratio=ratio,
            alpha=alpha,
            lr=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
        )
        out = Path(args.output_dir or results_cfg.get("root", "results/sae")) / f"tied_sae_alpha_{alpha:.2e}.pt"
        result.sae.save(out)
        print(f"Saved {out}")
        return

    alphas = [float(a) for a in sae_cfg.get("alphas", [1e-4, 3e-4, 8.6e-4, 1.4e-3, 3e-3])]
    output_dir = args.output_dir or Path(results_cfg.get("l1_sweep", "results/sae/l1_sweep"))
    rows = sweep_l1(
        dataset_dir,
        alphas=alphas,
        ratio=ratio,
        output_dir=output_dir,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device=device,
    )
    print(f"Sweep done: {len(rows)} models -> {output_dir}")


if __name__ == "__main__":
    main()
