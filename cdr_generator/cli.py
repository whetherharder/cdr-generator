"""CLI entrypoint for cdrgen commands."""

from __future__ import annotations

from pathlib import Path

import click

from cdr_generator.assets.generator import generate_all_assets
from cdr_generator.assets.store import (
    compute_config_hash,
    load_assets,
    save_assets,
    validate_assets as _validate_assets_fn,
)
from cdr_generator.config.loader import (
    ConfigLoadError,
    load_config,
    parse_override_strings,
    validate_config as _validate_config_fn,
)
from cdr_generator.engine.orchestrator import orchestrate
from cdr_generator.engine.runner import run_generation


@click.group()
@click.version_option(package_name="cdr-generator")
def main() -> None:
    """CDR Generator -- synthetic Call Detail Record generator for telecom testing."""


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
def validate_config(config_path: str) -> None:
    """Validate a YAML config file against the JSON schema."""
    errors = _validate_config_fn(Path(config_path))
    if errors:
        for err in errors:
            click.echo(err, err=True)
        raise SystemExit(1)
    click.echo("Config OK")


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
@click.option(
    "--assets-dir",
    default="./assets",
    type=click.Path(),
    help="Directory for generated assets.",
)
def generate_assets(config_path: str, assets_dir: str) -> None:
    """Generate cells, network elements, and subscribers from config."""
    cfg_path = Path(config_path)
    try:
        config = load_config(cfg_path)
    except (ConfigLoadError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    cells, nes, subs = generate_all_assets(config)
    config_hash = compute_config_hash(cfg_path)
    save_assets(Path(assets_dir), cells, nes, subs, config_hash)

    click.echo(
        f"Assets generated: {len(cells)} cells, {len(nes)} NEs, {len(subs)} subscribers"
    )
    click.echo(f"Saved to {assets_dir}/")


@main.command()
@click.option(
    "--assets-dir",
    default="./assets",
    type=click.Path(exists=True),
    help="Directory with generated assets.",
)
def validate_assets(assets_dir: str) -> None:
    """Validate generated assets for referential integrity."""
    try:
        cells, nes, subs, manifest = load_assets(Path(assets_dir))
    except (FileNotFoundError, Exception) as exc:
        raise click.ClickException(f"Failed to load assets: {exc}") from exc

    errors = _validate_assets_fn(cells, nes, subs)
    if errors:
        click.echo(f"Validation found {len(errors)} error(s):", err=True)
        for err in errors:
            click.echo(f"  - {err}", err=True)
        raise SystemExit(1)

    click.echo(
        f"Assets OK: {manifest.cell_count} cells, {manifest.ne_count} NEs, "
        f"{manifest.subscriber_count} subscribers"
    )


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
@click.option(
    "--assets-dir",
    default="./assets",
    type=click.Path(),
    help="Directory with generated assets.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for CDR files (default: from config).",
)
@click.option("--workers", default=1, type=int, help="Number of generator workers.")
@click.option("--dry-run", is_flag=True, help="Write CSV headers only, no records.")
@click.option("--progress", is_flag=True, help="Show progress bar.")
@click.option(
    "--override",
    multiple=True,
    help="Dot-path config override (e.g. subscribers.total_count=100).",
)
def generate(
    config_path: str,
    assets_dir: str,
    output_dir: str | None,
    workers: int,
    dry_run: bool,
    progress: bool,
    override: tuple[str, ...],
) -> None:
    """Generate synthetic CDR files."""
    cfg_path = Path(config_path)
    assets_path = Path(assets_dir)

    # Load config with overrides
    try:
        overrides = parse_override_strings(override) if override else None
        config = load_config(cfg_path, overrides=overrides)
    except (ConfigLoadError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    # Override output directory if specified via CLI
    if output_dir is not None:
        config.meta.output.path = output_dir

    # Load or generate assets
    config_hash = compute_config_hash(cfg_path)
    if assets_path.exists() and (assets_path / "manifest.json").exists():
        cells, nes, subs, manifest = load_assets(assets_path)
        if manifest.config_hash != config_hash:
            click.echo(
                "Config changed since last asset generation. Regenerating...", err=True
            )
            cells, nes, subs = generate_all_assets(config)
            save_assets(assets_path, cells, nes, subs, config_hash)
    else:
        cells, nes, subs = generate_all_assets(config)
        save_assets(assets_path, cells, nes, subs, config_hash)

    if workers > 1:
        click.echo(
            f"Warning: --workers {workers} requested but generation runs single-threaded "
            "(multi-process not yet implemented).",
            err=True,
        )
        stats = orchestrate(config, workers=workers, dry_run=dry_run)
    else:
        stats = run_generation(config, dry_run=dry_run)

    if dry_run:
        click.echo(
            f"Dry run: created {stats.files_written} header-only files in {config.meta.output.path}/"
        )
        return

    click.echo(
        f"Generation complete: {stats.total_records} records in {stats.files_written} files"
    )
    click.echo(
        f"  Voice: {stats.voice_records}, SMS: {stats.sms_records}, Data: {stats.data_records}"
    )
    if workers > 1:
        click.echo(f"  Workers: {workers}")
