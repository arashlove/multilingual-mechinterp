"""Placement 2 command line.

    python -m multilingual_mechinterp.placement2.cli <command> --config configs/placement_2.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from multilingual_mechinterp.placement2.experiment import (
    STAGES,
    RunContext,
    hash_inputs,
    load_placement2_config,
    pending_approvals,
)

DEFAULT_CONFIG = "configs/placement_2.yaml"

# Stage commands that exist in the CLI but are not implemented yet.
PLANNED: dict[str, str] = {
    "inspect-human-source": "human source contract (PR 1)",
    "preprocess-human": "human preprocessing (PR 2)",
    "summarize-human": "human preprocessing (PR 2)",
    "export-scenario-guide": "human preprocessing (PR 2)",
    "init-templates": "researcher scenario interface (PR 3)",
    "validate-data": "researcher scenario interface (PR 3)",
    "check-human-coverage": "researcher scenario interface (PR 3)",
    "finalize-human-labels": "researcher scenario interface (PR 3)",
    "run-baseline": "forced-choice baseline (PR 4)",
    "evaluate-human-agreement": "forced-choice baseline (PR 4)",
    "select-cases": "case selection and JLens (PR 5)",
    "fit-jlens": "case selection and JLens (PR 5)",
    "analyze-jlens": "case selection and JLens (PR 5)",
    "intervene": "intervention controls (PR 6)",
    "report": "reproducibility (PR 7)",
    "run-all": "reproducibility (PR 7)",
}


def _context(args: argparse.Namespace) -> RunContext:
    cfg = load_placement2_config(args.config)
    return RunContext(cfg, config_path=args.config, run_id=args.run_id)


def cmd_init_run(args: argparse.Namespace) -> int:
    ctx = _context(args)
    manifest = ctx.write_manifest()
    print(f"run directory: {ctx.run_dir}")
    print(f"manifest: {ctx.manifest_path}")
    _print_approvals(manifest["pending_approvals"])
    return 0


def cmd_check_config(args: argparse.Namespace) -> int:
    cfg = load_placement2_config(args.config)
    pending = pending_approvals(cfg)
    _print_approvals(pending)
    print("inputs:")
    for name, info in hash_inputs(cfg).items():
        print(f"  {name:<24} {'present' if info['exists'] else 'missing':<8} {info['path']}")
    return 1 if args.strict and pending else 0


def cmd_status(args: argparse.Namespace) -> int:
    ctx = _context(args)
    status = ctx.load_status()
    if not status:
        print(f"no status yet in {ctx.run_dir}; run `init-run` first")
        return 1
    for stage in STAGES:
        entry = status.get(stage, {"state": "pending"})
        detail = entry.get("error") or entry.get("reason") or ""
        print(f"{stage:<26} {entry['state']:<20} {detail}")
    return 0


def _planned(command: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        print(f"`{command}` is not implemented yet; planned in {PLANNED[command]}.", file=sys.stderr)
        return 2

    return handler


def _print_approvals(pending: list[str]) -> None:
    if not pending:
        print("all scientific decisions approved")
        return
    print("awaiting researcher approval (blocks a final scientific run):")
    for key in pending:
        print(f"  - {key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="placement2", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Callable[[argparse.Namespace], int], help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
        p.add_argument("--run-id", default=None, help="defaults to run.name in the config")
        p.set_defaults(handler=handler)
        return p

    add("init-run", cmd_init_run, "create the run folder, manifest and stage status")
    check = add("check-config", cmd_check_config, "list pending approvals and input files")
    check.add_argument("--strict", action="store_true", help="exit 1 if approvals are pending")
    add("status", cmd_status, "print stage status for a run")
    for command, owner in PLANNED.items():
        add(command, _planned(command), f"not implemented yet ({owner})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
