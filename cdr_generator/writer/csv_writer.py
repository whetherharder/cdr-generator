"""CSV+gzip writer producing one file per (NE, date), sorted by event_timestamp."""

from __future__ import annotations

import csv
import gzip
import io
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from cdr_generator.models.cdr import CDR_FIELDS, CDRRecord, to_csv_row

if TYPE_CHECKING:
    from cdr_generator.assets.models import NetworkElement
    from cdr_generator.config.models import CDRGeneratorConfig


class CSVWriter:
    """Writes CDR records to gzip-compressed CSV files.

    One file is created per ``(ne_id, date)`` combination using the pattern
    ``output_dir/ne_id/CDR_{ne_id}_{YYYYMMDD}.csv.gz``.
    """

    def __init__(
        self,
        output_dir: Path,
        filename_template: str = "CDR_{ne_id}_{date}.csv.gz",
        delimiter: str = ",",
        include_metadata: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._filename_template = filename_template
        self._delimiter = delimiter
        self._include_metadata = include_metadata

    def write_header(
        self,
        ne_id: str,
        date_str: str,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        """Create a gzip CSV file with an optional metadata comment and header row.

        Returns the path to the newly created file.
        """
        file_path = self._resolve_path(ne_id, date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wt", encoding="utf-8", newline="") as gz:
            if self._include_metadata and metadata:
                parts = ", ".join(f"{k}={v}" for k, v in metadata.items())
                gz.write(f"# {parts}\n")

            writer = csv.writer(gz, delimiter=self._delimiter)
            writer.writerow(CDR_FIELDS)

        return file_path

    def write_records(
        self,
        ne_id: str,
        date_str: str,
        records: list[CDRRecord],
    ) -> int:
        """Append *records* to the existing file for *(ne_id, date_str)*.

        Records are assumed to be pre-sorted by ``event_timestamp``.
        Returns the number of records written.
        """
        file_path = self._resolve_path(ne_id, date_str)

        with gzip.open(file_path, "at", encoding="utf-8", newline="") as gz:
            writer = csv.writer(gz, delimiter=self._delimiter)
            for record in records:
                writer.writerow(to_csv_row(record))

        return len(records)

    def _resolve_path(self, ne_id: str, date_str: str) -> Path:
        """Build the output file path for a given NE and date."""
        filename = self._filename_template.replace("{ne_id}", ne_id).replace(
            "{date}", date_str
        )
        return self._output_dir / ne_id / filename


class CsvWriter:
    """High-level CDR writer: one call to ``write_file`` creates a complete
    gzip-compressed CSV for a given ``(ne_id, date)`` pair.

    Usage::

        writer = CsvWriter(output_dir=Path("./output"))
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()
    """

    def __init__(
        self,
        output_dir: str | Path,
        delimiter: str = ",",
        include_metadata: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._delimiter = delimiter
        self._include_metadata = include_metadata

    def write_file(
        self,
        ne_id: str,
        file_date: date,
        records: list[CDRRecord] | None = None,
    ) -> Path:
        """Write a complete CDR file for *(ne_id, file_date)*.

        Creates the NE subdirectory, writes an optional metadata comment,
        the CSV header, and any *records*.  Returns the file path.
        """
        date_str = file_date.strftime("%Y%m%d")
        file_path = self._output_dir / ne_id / f"CDR_{ne_id}_{date_str}.csv.gz"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wt", encoding="utf-8", newline="") as gz:
            if self._include_metadata:
                gz.write(f"# ne_id={ne_id}, date={date_str}\n")

            writer = csv.writer(gz, delimiter=self._delimiter)
            writer.writerow(CDR_FIELDS)

            for record in records or []:
                writer.writerow(to_csv_row(record))

        return file_path

    def close(self) -> None:
        """No-op -- each ``write_file`` call is self-contained."""

    def __enter__(self) -> CsvWriter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def create_empty_output(
    config: CDRGeneratorConfig,
    network_elements: list[NetworkElement],
) -> list[Path]:
    """Create header-only CSV+gzip files for every (NE, day) in the time range.

    Returns the list of created file paths.
    """
    output_cfg = config.meta.output
    output_dir = Path(output_cfg.path)

    writer = CSVWriter(
        output_dir=output_dir,
        filename_template=output_cfg.filename_template,
        delimiter=output_cfg.csv_delimiter,
        include_metadata=output_cfg.include_metadata_comment,
    )

    start_date = config.meta.time_range.start.date()
    end_date = config.meta.time_range.end.date()
    dates = _date_range(start_date, end_date)

    created: list[Path] = []
    for ne in network_elements:
        for d in dates:
            date_str = d.strftime("%Y%m%d")
            metadata = {"ne_id": ne.id, "ne_type": ne.ne_type}
            path = writer.write_header(ne.id, date_str, metadata=metadata)
            created.append(path)

    return created


def _date_range(start: date, end: date) -> list[date]:
    """Generate a list of dates from *start* to *end* inclusive."""
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates
