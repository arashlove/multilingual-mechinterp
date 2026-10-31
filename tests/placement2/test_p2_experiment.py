from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from multilingual_mechinterp.placement2.experiment import (
    STAGES,
    RunContext,
    load_placement2_config,
    pending_approvals,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "placement_2.yaml"


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    cfg = load_placement2_config(CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg["run"]["output_root"] = "out"
    cfg["model"]["local_path"] = str(tmp_path / "no_model_here")
    return cfg


def test_repo_config_lists_unsigned_decisions():
    pending = pending_approvals(load_placement2_config(CONFIG))
    assert "human.primary_reference_aggregation" in pending
    assert "human.min_matched_responses_per_culture" in pending
    assert "jlens.source_layers" not in pending
    assert "model.revision" in pending


def test_pending_clears_when_signed(cfg):
    cfg["human"]["primary_reference_aggregation"] = "equal_country_vote"
    cfg["human"]["min_matched_responses_per_culture"] = 30
    cfg["jlens"]["source_layers"] = [22, 26, 30, 38]
    cfg["model"]["revision"] = "0" * 40
    assert pending_approvals(cfg) == []


def test_missing_section_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run: {name: x, output_root: out}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="human"):
        load_placement2_config(path)


def test_manifest_records_inputs_and_initialises_status(cfg, tmp_path):
    inputs = tmp_path / "data" / "placement_2" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "concepts.json").write_text('{"status": "draft"}', encoding="utf-8")

    ctx = RunContext(cfg, run_id="r1", root=tmp_path)
    manifest = ctx.write_manifest()

    assert ctx.manifest_path.is_file()
    assert manifest["inputs"]["concepts"]["exists"] is True
    assert len(manifest["inputs"]["concepts"]["sha256"]) == 64
    assert manifest["inputs"]["questions_fa"] == {
        "path": cfg["inputs"]["questions_fa"],
        "exists": False,
        "sha256": None,
    }
    assert "numpy" in manifest["environment"]["packages"]
    assert "matplotlib" in manifest["environment"]["packages"]
    assert manifest["model"]["local_snapshot"] is None
    status = json.loads(ctx.status_path.read_text(encoding="utf-8"))
    assert list(status) == list(STAGES)
    assert all(entry["state"] == "pending" for entry in status.values())


def test_manifest_hashes_local_model_snapshot(cfg, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    cfg["model"]["local_path"] = str(model_dir)
    manifest = RunContext(cfg, run_id="r1", root=tmp_path).build_manifest()
    assert list(manifest["model"]["local_snapshot"]) == ["config.json"]


def test_stage_ok_and_freshness(cfg, tmp_path):
    ctx = RunContext(cfg, run_id="r1", root=tmp_path)
    with ctx.stage("run_baseline", fingerprint="abc") as out_dir:
        (out_dir / "result.json").write_text("{}", encoding="utf-8")

    entry = ctx.load_status()["run_baseline"]
    assert entry["state"] == "ok"
    assert entry["wall_seconds"] >= 0
    assert ctx.is_fresh("run_baseline", "abc")
    assert not ctx.is_fresh("run_baseline", "changed-input")


def test_stage_failure_is_recorded_and_reraised(cfg, tmp_path):
    ctx = RunContext(cfg, run_id="r1", root=tmp_path)
    with pytest.raises(RuntimeError), ctx.stage("fit_jlens", fingerprint="abc"):
        raise RuntimeError("out of memory")

    entry = ctx.load_status()["fit_jlens"]
    assert entry["state"] == "failed"
    assert "out of memory" in entry["error"]
    assert not ctx.is_fresh("fit_jlens", "abc")


def test_block_and_unknown_stage(cfg, tmp_path):
    ctx = RunContext(cfg, run_id="r1", root=tmp_path)
    ctx.block("preprocess_human", "raw SharedResponses file not supplied")
    assert ctx.load_status()["preprocess_human"]["state"] == "block_missing_data"
    with pytest.raises(ValueError):
        ctx.set_status("not_a_stage", "ok")


def test_stage_never_deletes_existing_checkpoints(cfg, tmp_path):
    ctx = RunContext(cfg, run_id="r1", root=tmp_path)
    lens = ctx.stage_dir("fit_jlens") / "lens.pt"
    lens.write_bytes(b"weights")
    with ctx.stage("fit_jlens", fingerprint="abc"):
        pass
    ctx.write_manifest()
    assert lens.is_file()
