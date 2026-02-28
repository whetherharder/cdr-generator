"""Tests for asset generation, storage, and validation.

Covers PM Task #16 and plan.md Phase 1 acceptance criteria:
- Assets: cells, NE, subscribers on disk, manifest with config hash
- Validation: cell_id in cells, serving_ne in NE, etc.
- Generate-assets creates expected files on disk
- Subscriber count matches config total_count
"""

import json
import pathlib

import pytest

from cdr_generator.assets.models import AssetManifest, Cell, NetworkElement, Subscriber
from cdr_generator.assets.generator import (
    generate_all_assets,
    generate_cells,
    generate_network_elements,
    generate_subscribers,
)
from cdr_generator.assets.store import (
    compute_config_hash,
    load_assets,
    save_assets,
    validate_assets,
)
from cdr_generator.config.loader import load_config


# ============================================================================
# Asset Models
# ============================================================================


class TestAssetModels:
    """Test that asset Pydantic models are properly defined."""

    def test_cell_model_importable(self) -> None:
        assert Cell is not None

    def test_network_element_model_importable(self) -> None:
        assert NetworkElement is not None

    def test_subscriber_model_importable(self) -> None:
        assert Subscriber is not None

    def test_manifest_model_importable(self) -> None:
        assert AssetManifest is not None

    def test_cell_model_fields(self) -> None:
        cell = Cell(
            cell_id=20011,
            tac=2001,
            ecgi="25001-20011",
            lat=55.7558,
            lon=37.6173,
            azimuth=0,
            sector=1,
            cell_type="urban",
            capacity="high",
            neighbors=[20012, 20013],
        )
        assert cell.cell_id == 20011
        assert cell.tac == 2001
        assert cell.cell_type == "urban"
        assert cell.capacity == "high"
        assert cell.neighbors == [20012, 20013]

    def test_network_element_model_fields(self) -> None:
        ne = NetworkElement(
            id="msc-01",
            ne_type="msc",
            vendor="ericsson",
            address="10.0.1.1",
            serves_tacs=[2001, 2002],
            extensions_template={"ericsson.sw_version": "R16B"},
            produces=["mo_call", "mt_call"],
        )
        assert ne.id == "msc-01"
        assert ne.ne_type == "msc"
        assert ne.vendor == "ericsson"
        assert ne.serves_tacs == [2001, 2002]

    def test_subscriber_model_fields(self) -> None:
        sub = Subscriber(
            imsi="250010000000001",
            msisdn="+79160000001",
            imei="353325080000001",
            profile_name="office_worker",
            home_cell_id=20011,
            work_cell_id=20012,
            serving_ne_id="msc-01",
        )
        assert sub.imsi == "250010000000001"
        assert sub.msisdn == "+79160000001"
        assert sub.profile_name == "office_worker"
        assert sub.home_cell_id == 20011
        assert sub.work_cell_id == 20012
        assert sub.serving_ne_id == "msc-01"

    def test_subscriber_work_cell_optional(self) -> None:
        """work_cell_id can be None (e.g., retiree profile)."""
        sub = Subscriber(
            imsi="250010000000002",
            msisdn="+79160000002",
            imei="353325080000002",
            profile_name="retiree",
            home_cell_id=20011,
            work_cell_id=None,
            serving_ne_id="msc-01",
        )
        assert sub.work_cell_id is None


# ============================================================================
# Asset Generation
# ============================================================================


class TestAssetGeneration:
    """Test that generate_all_assets produces correct data."""

    def test_generate_all_assets_returns_tuple(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """generate_all_assets returns (cells, NEs, subscribers) tuple."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)

        assert isinstance(cells, list)
        assert isinstance(nes, list)
        assert isinstance(subscribers, list)
        assert len(cells) > 0
        assert len(nes) > 0
        assert len(subscribers) > 0

    def test_generated_cell_count_matches_config(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Number of generated cells should match inline items in config."""
        config = load_config(sample_config_path)
        cells, _, _ = generate_all_assets(config)
        assert len(cells) == len(config.network.cells.items)

    def test_generated_ne_count_matches_config(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Number of NEs should match config elements."""
        config = load_config(sample_config_path)
        _, nes, _ = generate_all_assets(config)
        assert len(nes) == len(config.network.elements)

    def test_generated_subscriber_count_matches_config(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Number of subscribers should match config total_count."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        assert len(subscribers) == config.subscribers.total_count

    def test_generated_cells_have_correct_types(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """All generated cells should be Cell instances."""
        config = load_config(sample_config_path)
        cells, _, _ = generate_all_assets(config)
        for cell in cells:
            assert isinstance(cell, Cell)

    def test_generated_nes_have_correct_types(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """All generated NEs should be NetworkElement instances."""
        config = load_config(sample_config_path)
        _, nes, _ = generate_all_assets(config)
        for ne in nes:
            assert isinstance(ne, NetworkElement)

    def test_generated_subscribers_have_correct_types(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """All generated subscribers should be Subscriber instances."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        for sub in subscribers:
            assert isinstance(sub, Subscriber)

    def test_generation_is_deterministic(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Same config/seed produces identical assets."""
        config = load_config(sample_config_path)
        cells1, nes1, subs1 = generate_all_assets(config)
        cells2, nes2, subs2 = generate_all_assets(config)

        assert len(subs1) == len(subs2)
        for s1, s2 in zip(subs1, subs2):
            assert s1.imsi == s2.imsi
            assert s1.home_cell_id == s2.home_cell_id
            assert s1.profile_name == s2.profile_name

    def test_subscriber_imsi_prefix(self, sample_config_path: pathlib.Path) -> None:
        """All subscriber IMSIs should start with the configured prefix."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        for sub in subscribers:
            assert sub.imsi.startswith(config.subscribers.imsi_prefix), (
                f"IMSI {sub.imsi} does not start with {config.subscribers.imsi_prefix}"
            )

    def test_subscriber_msisdn_prefix(self, sample_config_path: pathlib.Path) -> None:
        """All subscriber MSISDNs should start with the configured prefix."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        for sub in subscribers:
            assert sub.msisdn.startswith(config.subscribers.msisdn_prefix), (
                f"MSISDN {sub.msisdn} does not start with {config.subscribers.msisdn_prefix}"
            )

    def test_subscriber_imsi_uniqueness(self, sample_config_path: pathlib.Path) -> None:
        """All subscriber IMSIs must be unique."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        imsis = [s.imsi for s in subscribers]
        assert len(imsis) == len(set(imsis)), "Duplicate IMSIs found"

    def test_subscriber_profiles_from_config(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """All subscriber profile_name values must reference a config profile."""
        config = load_config(sample_config_path)
        _, _, subscribers = generate_all_assets(config)
        valid_profiles = {p.name for p in config.subscribers.profiles}
        for sub in subscribers:
            assert sub.profile_name in valid_profiles, (
                f"Subscriber {sub.imsi} has unknown profile: {sub.profile_name}"
            )


# ============================================================================
# Asset Save / Load (Store)
# ============================================================================


class TestAssetStore:
    """Test save_assets and load_assets roundtrip."""

    def test_save_creates_files(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """save_assets must create cells.json, network_elements.json, subscribers.json, manifest.json."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)

        expected_files = {
            "cells.json",
            "network_elements.json",
            "subscribers.json",
            "manifest.json",
        }
        actual_files = {f.name for f in tmp_assets_dir.iterdir()}
        assert expected_files <= actual_files, (
            f"Missing files. Expected: {expected_files}, Got: {actual_files}"
        )

    def test_load_roundtrip_cells(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """Cells survive save/load roundtrip."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)
        loaded_cells, _, _, _ = load_assets(tmp_assets_dir)

        assert len(loaded_cells) == len(cells)
        for orig, loaded in zip(cells, loaded_cells):
            assert orig.cell_id == loaded.cell_id
            assert orig.tac == loaded.tac

    def test_load_roundtrip_subscribers(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """Subscribers survive save/load roundtrip with correct count."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)
        _, _, loaded_subs, _ = load_assets(tmp_assets_dir)

        assert len(loaded_subs) == config.subscribers.total_count

    def test_load_from_empty_dir_raises(self, tmp_assets_dir: pathlib.Path) -> None:
        """Loading from an empty directory must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_assets(tmp_assets_dir)


# ============================================================================
# Asset Manifest
# ============================================================================


class TestAssetManifest:
    """Test that manifest correctly tracks generation metadata."""

    def test_manifest_contains_config_hash(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """manifest.json must contain config_hash matching the file hash."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)

        manifest_path = tmp_assets_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json must exist"

        manifest = json.loads(manifest_path.read_text())
        assert manifest["config_hash"] == config_hash
        assert len(manifest["config_hash"]) > 0

    def test_manifest_contains_counts(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """Manifest records correct counts of each asset type."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)

        manifest_path = tmp_assets_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest["cell_count"] == len(cells)
        assert manifest["ne_count"] == len(nes)
        assert manifest["subscriber_count"] == len(subscribers)
        assert manifest["subscriber_count"] == config.subscribers.total_count

    def test_manifest_contains_generated_at(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """Manifest records generation timestamp."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)

        manifest_path = tmp_assets_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "generated_at" in manifest
        assert len(str(manifest["generated_at"])) > 0

    def test_manifest_loads_as_model(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
    ) -> None:
        """Manifest can be loaded via load_assets and is an AssetManifest."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        config_hash = compute_config_hash(sample_config_path)

        save_assets(tmp_assets_dir, cells, nes, subscribers, config_hash)
        _, _, _, manifest = load_assets(tmp_assets_dir)

        assert isinstance(manifest, AssetManifest)
        assert manifest.config_hash == config_hash
        assert manifest.subscriber_count == len(subscribers)


# ============================================================================
# Asset Validation (Referential Integrity)
# ============================================================================


class TestAssetValidation:
    """Test validate_assets catches referential integrity issues."""

    def test_validate_passes_on_valid_assets(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Freshly generated assets must pass validation (zero errors)."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)

        errors = validate_assets(cells, nes, subscribers)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_subscriber_serving_ne_valid(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Every subscriber's serving_ne_id must be in NE id set."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        ne_ids = {ne.id for ne in nes}

        for sub in subscribers:
            assert sub.serving_ne_id in ne_ids, (
                f"Subscriber {sub.imsi}: serving_ne_id={sub.serving_ne_id} "
                f"not in {ne_ids}"
            )

    def test_subscriber_home_cell_valid(self, sample_config_path: pathlib.Path) -> None:
        """Every subscriber's home_cell_id must be in cell id set."""
        config = load_config(sample_config_path)
        cells, nes, subscribers = generate_all_assets(config)
        cell_ids = {c.cell_id for c in cells}

        for sub in subscribers:
            assert sub.home_cell_id in cell_ids, (
                f"Subscriber {sub.imsi}: home_cell_id={sub.home_cell_id} "
                f"not in {cell_ids}"
            )

    def test_cell_neighbors_valid(self, sample_config_path: pathlib.Path) -> None:
        """Every cell's neighbor must reference an existing cell_id."""
        config = load_config(sample_config_path)
        cells, _, _ = generate_all_assets(config)
        cell_ids = {c.cell_id for c in cells}

        for cell in cells:
            for neighbor_id in cell.neighbors:
                assert neighbor_id in cell_ids, (
                    f"Cell {cell.cell_id}: neighbor {neighbor_id} not in {cell_ids}"
                )

    def test_validate_detects_bad_serving_ne(self) -> None:
        """validate_assets should catch a subscriber referencing a nonexistent NE."""
        cells = [
            Cell(
                cell_id=1,
                tac=100,
                ecgi="x-1",
                lat=0,
                lon=0,
                azimuth=0,
                sector=1,
                cell_type="urban",
                capacity="high",
            )
        ]
        nes = [
            NetworkElement(
                id="msc-01", ne_type="msc", vendor="ericsson", serves_tacs=[100]
            )
        ]
        subs = [
            Subscriber(
                imsi="250010",
                msisdn="+79160",
                imei="000000000000000",
                profile_name="test",
                home_cell_id=1,
                serving_ne_id="nonexistent",
            )
        ]

        errors = validate_assets(cells, nes, subs)
        assert len(errors) > 0
        assert any("nonexistent" in e for e in errors)

    def test_validate_detects_bad_home_cell(self) -> None:
        """validate_assets should catch a subscriber referencing a nonexistent home cell."""
        cells = [
            Cell(
                cell_id=1,
                tac=100,
                ecgi="x-1",
                lat=0,
                lon=0,
                azimuth=0,
                sector=1,
                cell_type="urban",
                capacity="high",
            )
        ]
        nes = [
            NetworkElement(
                id="msc-01", ne_type="msc", vendor="ericsson", serves_tacs=[100]
            )
        ]
        subs = [
            Subscriber(
                imsi="250010",
                msisdn="+79160",
                imei="000000000000000",
                profile_name="test",
                home_cell_id=999,
                serving_ne_id="msc-01",
            )
        ]

        errors = validate_assets(cells, nes, subs)
        assert len(errors) > 0
        assert any("999" in e for e in errors)

    def test_validate_detects_bad_neighbor(self) -> None:
        """validate_assets should catch a cell referencing a nonexistent neighbor."""
        cells = [
            Cell(
                cell_id=1,
                tac=100,
                ecgi="x-1",
                lat=0,
                lon=0,
                azimuth=0,
                sector=1,
                cell_type="urban",
                capacity="high",
                neighbors=[999],
            )
        ]
        nes = [
            NetworkElement(
                id="msc-01", ne_type="msc", vendor="ericsson", serves_tacs=[100]
            )
        ]
        subs: list[Subscriber] = []

        errors = validate_assets(cells, nes, subs)
        assert len(errors) > 0
        assert any("999" in e for e in errors)
