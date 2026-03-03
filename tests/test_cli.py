"""Tests for CLI entrypoint.

Acceptance criteria from plan.md Phase 1:
- CLI commands: validate-config, generate-assets, validate-assets, generate
- generate --dry-run produces header-only files
- generate-assets creates assets directory with manifest
- validate-config reports errors for invalid config
"""

import json
import pathlib

import pytest
from click.testing import CliRunner

from cdr_generator.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Click test runner."""
    return CliRunner()


# ===========================================================================
# validate-config command
# ===========================================================================


class TestValidateConfigCommand:
    """Test the validate-config CLI command."""

    def test_valid_config_exits_zero(
        self, runner: CliRunner, sample_config_path: pathlib.Path
    ) -> None:
        result = runner.invoke(
            main, ["validate-config", "--config", str(sample_config_path)]
        )
        assert result.exit_code == 0
        assert "Config OK" in result.output

    def test_missing_config_file(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        missing = tmp_path / "nonexistent.yaml"
        result = runner.invoke(main, ["validate-config", "--config", str(missing)])
        assert result.exit_code != 0

    def test_invalid_yaml_config(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("  bad:\n yaml: [broken\n", encoding="utf-8")
        result = runner.invoke(main, ["validate-config", "--config", str(bad_config)])
        assert result.exit_code != 0

    def test_empty_config_fails(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        empty_cfg = tmp_path / "empty.yaml"
        empty_cfg.write_text("{}", encoding="utf-8")
        result = runner.invoke(main, ["validate-config", "--config", str(empty_cfg)])
        assert result.exit_code != 0


# ===========================================================================
# generate-assets command
# ===========================================================================


class TestGenerateAssetsCommand:
    """Test the generate-assets CLI command."""

    def test_generate_assets_creates_directory(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate-assets",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
            ],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert assets_dir.exists()

    def test_generate_assets_creates_manifest(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate-assets",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
            ],
        )
        assert result.exit_code == 0
        manifest_path = assets_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "config_hash" in manifest
        assert "cell_count" in manifest
        assert "ne_count" in manifest
        assert "subscriber_count" in manifest

    def test_generate_assets_creates_json_files(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        assets_dir = tmp_path / "assets"
        runner.invoke(
            main,
            [
                "generate-assets",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
            ],
        )
        for name in (
            "cells.json",
            "network_elements.json",
            "subscribers.json",
            "manifest.json",
        ):
            assert (assets_dir / name).exists(), f"{name} not created"

    def test_generate_assets_output_message(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate-assets",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Assets generated:" in result.output
        assert "cells" in result.output
        assert "NEs" in result.output
        assert "subscribers" in result.output


# ===========================================================================
# validate-assets command
# ===========================================================================


class TestValidateAssetsCommand:
    """Test the validate-assets CLI command."""

    def test_validate_valid_assets(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        assets_dir = tmp_path / "assets"
        # First generate assets
        runner.invoke(
            main,
            [
                "generate-assets",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
            ],
        )
        # Then validate
        result = runner.invoke(
            main, ["validate-assets", "--assets-dir", str(assets_dir)]
        )
        assert result.exit_code == 0
        assert "Assets OK" in result.output

    def test_validate_missing_assets_dir(
        self, runner: CliRunner, tmp_path: pathlib.Path
    ) -> None:
        missing_dir = tmp_path / "no_assets"
        result = runner.invoke(
            main, ["validate-assets", "--assets-dir", str(missing_dir)]
        )
        assert result.exit_code != 0


# ===========================================================================
# generate command (dry-run)
# ===========================================================================


class TestGenerateCommand:
    """Test the generate CLI command."""

    def test_generate_dry_run(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, f"dry-run failed: {result.output}"
        assert "Dry run" in result.output
        assert "header-only" in result.output

    def test_generate_dry_run_creates_files(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        # Check that at least one .csv.gz file was created
        gz_files = list(output_dir.rglob("*.csv.gz"))
        assert len(gz_files) > 0, "No .csv.gz files created by dry-run"

    def test_generate_dry_run_files_are_header_only(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        import gzip

        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        gz_files = list(output_dir.rglob("*.csv.gz"))
        for gz_file in gz_files:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                lines = f.readlines()
            data_lines = [line for line in lines if not line.startswith("#")]
            assert len(data_lines) == 1, (
                f"File {gz_file.name} has {len(data_lines)} data lines, expected 1 (header only)"
            )

    def test_generate_without_dry_run_produces_records(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--override",
                "subscribers.total_count=10",
                "--override",
                "meta.time_range.end=2025-01-01T00:59:59Z",
            ],
        )
        assert result.exit_code == 0
        assert "Generation complete" in result.output

    def test_generate_with_override(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        result = runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--override",
                "subscribers.total_count=50",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_generate_auto_regenerates_assets_on_config_change(
        self,
        runner: CliRunner,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """If config hash changes, generate should regenerate assets."""
        output_dir = tmp_path / "output"
        assets_dir = tmp_path / "assets"
        # First run creates assets
        runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert (assets_dir / "manifest.json").exists()

        # Tamper the manifest hash to simulate config change
        manifest = json.loads((assets_dir / "manifest.json").read_text())
        manifest["config_hash"] = "stale_hash"
        (assets_dir / "manifest.json").write_text(json.dumps(manifest))

        # Second run should detect mismatch and regenerate
        result = runner.invoke(
            main,
            [
                "generate",
                "--config",
                str(sample_config_path),
                "--assets-dir",
                str(assets_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0


# ===========================================================================
# CLI help and version
# ===========================================================================


class TestCLIHelp:
    """Test CLI help and version output."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "CDR Generator" in result.output

    def test_main_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_generate_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--override" in result.output
        assert "--config" in result.output

    def test_validate_config_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["validate-config", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
