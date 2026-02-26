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
    """Per-file context manager writer for a single (ne_id, date) combination.

    Usage::

        with CsvWriter(output_dir, ne_id, day, delimiter=",") as w:
            w.write_header(CDR_FIELDS)
            w.write_record({"record_type": "mo_call", ...})
    """

    def __init__(
        self,
        output_dir: str | Path,
        ne_id: str,
        date: date,
        delimiter: str = ",",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._ne_id = ne_id
        self._date = date
        self._delimiter = delimiter
        self._file_path = (
            self._output_dir
            / ne_id
            / f"CDR_{ne_id}_{date:%Y%m%d}.csv.gz"
        )
        self._gz_file: gzip.GzipFile | None = None
        self._writer: csv.DictWriter | None = None
        self._text_wrapper: io.TextIOWrapper | None = None

    @property
    def file_path(self) -> Path:
        return self._file_path

    def open(self) -> CsvWriter:
        """Open the gzip file for writing."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._gz_file = gzip.open(self._file_path, "wb")
        self._text_wrapper = io.TextIOWrapper(self._gz_file, encoding="utf-8", newline="")
        return self

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._text_wrapper is not None:
            self._text_wrapper.close()
            self._text_wrapper = None
        if self._gz_file is not None:
            self._gz_file = None
        self._writer = None

    def write_header(self, columns: list[str]) -> None:
        """Write the CSV header row and initialize the DictWriter."""
        if self._text_wrapper is None:
            self.open()
        self._writer = csv.DictWriter(
            self._text_wrapper,  # type: ignore[arg-type]
            fieldnames=columns,
            delimiter=self._delimiter,
            extrasaction="ignore",
        )
        self._writer.writeheader()

    def write_record(self, row: dict[str, object]) -> None:
        """Write a single record row (dict keyed by column name)."""
        if self._writer is None:
            raise RuntimeError("write_header() must be called before write_record()")
        self._writer.writerow(row)

    def __enter__(self) -> CsvWriter:
        return self.open()

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
