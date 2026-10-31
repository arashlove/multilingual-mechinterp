from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multilingual_mechinterp.placement2 import cli

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "placement_2.yaml"


@pytest.fixture()
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["model"]["local_path"] = str(tmp_path / "no_model_here")
    path = tmp_path / "placement_2.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return path


def test_init_run_then_status(config_path, tmp_path, capsys):
    assert cli.main(["init-run", "--config", str(config_path), "--run-id", "smoke"]) == 0
    run_dir = tmp_path / "output" / "placement_2" / "smoke"
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "status.json").is_file()
    assert "model.revision" in capsys.readouterr().out

    assert cli.main(["status", "--config", str(config_path), "--run-id", "smoke"]) == 0
    assert "preprocess_human" in capsys.readouterr().out


def test_check_config_strict_fails_while_approvals_pending(config_path):
    assert cli.main(["check-config", "--config", str(config_path)]) == 0
    assert cli.main(["check-config", "--config", str(config_path), "--strict"]) == 1


def test_status_without_run(config_path):
    assert cli.main(["status", "--config", str(config_path), "--run-id", "none"]) == 1


@pytest.mark.parametrize("command", sorted(cli.PLANNED))
def test_planned_commands_exit_cleanly(command, config_path, capsys):
    assert cli.main([command, "--config", str(config_path)]) == 2
    assert "not implemented yet" in capsys.readouterr().err
