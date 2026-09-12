#!/usr/bin/env python
"""Extract residual activations for SAE training (Phase 0)."""

from __future__ import annotations

import argparse
from pathlib import Path

from multilingual_mechinterp.sae import extract_and_save
from multilingual_mechinterp.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pythia70m_sae.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--n-texts", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.exists() else {}
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    metas = extract_and_save(
        model_name=args.model or model_cfg.get("name", "EleutherAI/pythia-70m-deduped"),
        dataset_name=args.dataset or data_cfg.get("dataset", "NeelNanda/pile-10k"),
        n_texts=args.n_texts or int(data_cfg.get("n_texts", 256)),
        layer=args.layer if args.layer is not None else int(data_cfg.get("layer", 1)),
        output_dir=args.output_dir
        or Path(data_cfg.get("activations_dir", "data/processed/activations/layer_1")),
        max_length=int(data_cfg.get("max_length", 128)),
        batch_size=args.batch_size,
        device=args.device or model_cfg.get("device"),
    )
    total = sum(m.n_tokens for m in metas)
    print(f"Wrote {len(metas)} chunk(s), {total} tokens -> {metas[0].path.parent}")


if __name__ == "__main__":
    main()
