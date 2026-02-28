"""Pydantic v2 models mirroring the CDR Generator YAML config structure."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NEType(str, Enum):
    msc = "msc"
    sgw = "sgw"
    pgw = "pgw"
    smsc = "smsc"


class Vendor(str, Enum):
    ericsson = "ericsson"
    huawei = "huawei"
    nokia = "nokia"


class CellType(str, Enum):
    urban = "urban"
    suburban = "suburban"
    rural = "rural"


class CellCapacity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


# ---------------------------------------------------------------------------
# Shared / reusable models
# ---------------------------------------------------------------------------


class Distribution(BaseModel):
    """Statistical distribution descriptor: {type: <name>, params: {…}}."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class TimeRange(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _end_after_start(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("time_range.end must be after time_range.start")
        return self


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    format: str = "csv_gzip"
    path: str = "./output"
    filename_template: str = "CDR_{ne_id}_{date}.csv.gz"
    csv_delimiter: str = ","
    include_metadata_comment: bool = True


class ParallelismConfig(BaseModel):
    workers: int = Field(default=16, ge=1)


class MetaConfig(BaseModel):
    seed: int = 42
    time_range: TimeRange
    time_step_seconds: int = Field(default=60, ge=1)
    output: OutputConfig = Field(default_factory=OutputConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


class OperatorConfig(BaseModel):
    mcc: str
    mnc: str
    name: str


class NetworkElementConfig(BaseModel):
    id: str
    type: NEType
    vendor: Vendor
    address: str = ""
    serves_tacs: list[int] = Field(default_factory=list)
    serves_apns: list[str] = Field(default_factory=list)
    extensions_template: dict[str, Any] = Field(default_factory=dict)
    produces: list[str] = Field(default_factory=list)


class CellItemConfig(BaseModel):
    cell_id: int
    tac: int
    ecgi: str
    lat: float
    lon: float
    azimuth: int = Field(ge=0, lt=360)
    sector: int = Field(ge=1)
    type: CellType
    capacity: CellCapacity
    neighbors: list[int] = Field(default_factory=list)


class AutoGenerateConfig(BaseModel):
    enabled: bool = False
    num_cells: int = 200
    center_lat: float = 0.0
    center_lon: float = 0.0
    radius_km: float = 20.0
    urban_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    suburban_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    neighbor_radius_km: float = 2.0
    sectors_per_site: int = Field(default=3, ge=1)


class CellsConfig(BaseModel):
    source: str = "inline"
    file_path: str | None = None
    items: list[CellItemConfig] = Field(default_factory=list)


class NetworkConfig(BaseModel):
    operator: OperatorConfig
    elements: list[NetworkElementConfig] = Field(default_factory=list)
    cells: CellsConfig = Field(default_factory=CellsConfig)
    auto_generate: AutoGenerateConfig = Field(default_factory=AutoGenerateConfig)


# ---------------------------------------------------------------------------
# subscribers
# ---------------------------------------------------------------------------


class DailyRates(BaseModel):
    mo_call: Distribution
    mo_sms: Distribution
    data_session: Distribution
    mt_call: Distribution
    mt_sms: Distribution


class HourlyWeights(BaseModel):
    voice: list[float] = Field(min_length=24, max_length=24)
    data: list[float] = Field(min_length=24, max_length=24)
    sms: list[float] = Field(min_length=24, max_length=24)


class DayOfWeekMultipliers(BaseModel):
    voice: list[float] = Field(min_length=7, max_length=7)
    data: list[float] = Field(min_length=7, max_length=7)
    sms: list[float] = Field(min_length=7, max_length=7)


class MobilityConfig(BaseModel):
    home_cell_strategy: str
    work_cell_strategy: str | None = None
    commute_hours: list[int] = Field(default_factory=list)
    roaming_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    handover_during_call: float = Field(default=0.0, ge=0.0, le=1.0)


class SubscriberProfile(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    description: str = ""
    imei_tac_pool: list[str] = Field(default_factory=list)
    rat_preference: list[str] = Field(default_factory=list)
    daily_rates: DailyRates
    hourly_weights: HourlyWeights
    day_of_week_multipliers: DayOfWeekMultipliers
    mobility: MobilityConfig


class ContactBookConfig(BaseModel):
    avg_contacts: int = 15
    degree_distribution: Distribution = Field(
        default_factory=lambda: Distribution(
            type="zipf", params={"a": 2.0, "min": 3, "max": 100}
        )
    )
    asymmetric: bool = True
    intra_profile_bias: float = 1.5
    repeat_call_probability: float = Field(default=0.6, ge=0.0, le=1.0)
    external_call_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class ExternalNumberPrefix(BaseModel):
    prefix: str
    weight: float = Field(ge=0.0)
    label: str = ""


class ExternalNumbersConfig(BaseModel):
    count: int = 50000
    prefixes: list[ExternalNumberPrefix] = Field(default_factory=list)


class SubscribersConfig(BaseModel):
    total_count: int = Field(ge=1)
    imsi_prefix: str = "250010"
    msisdn_prefix: str = "+7916"
    profiles: list[SubscriberProfile] = Field(min_length=1)
    contact_book: ContactBookConfig = Field(default_factory=ContactBookConfig)
    external_numbers: ExternalNumbersConfig = Field(
        default_factory=ExternalNumbersConfig
    )


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


class CauseWeight(BaseModel):
    """Weighted cause code entry used in voice failure/termination lists."""

    cause: str
    code: int | None = None
    weight: float = Field(ge=0.0)


class DistributionWithBounds(BaseModel):
    """Distribution with optional min/max clamps."""

    distribution: Distribution
    min_seconds: float | None = None
    max_seconds: float | None = None
    min_bytes: int | None = None


class VoiceEventConfig(BaseModel):
    success_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    failure_causes: list[CauseWeight] = Field(default_factory=list)
    duration: DistributionWithBounds
    setup_duration_ms: DistributionWithBounds | None = None
    call_forwarding_rate: float = Field(default=0.03, ge=0.0, le=1.0)
    normal_termination_causes: list[CauseWeight] = Field(default_factory=list)
    paired_timestamp_jitter_ms: int = 1000


class VolumeMultiplier(BaseModel):
    uplink: float = 1.0
    downlink: float = 1.0


class QoSEntry(BaseModel):
    qci: int
    weight: float = Field(ge=0.0)
    label: str = ""


class PartialRecordsConfig(BaseModel):
    enabled: bool = True
    max_record_duration_seconds: int = 3600
    max_record_volume_bytes: int = 104857600


class DataTerminationCause(BaseModel):
    cause: str
    weight: float = Field(ge=0.0)


class DataEventConfig(BaseModel):
    duration: DistributionWithBounds
    volume_uplink: DistributionWithBounds
    volume_downlink: DistributionWithBounds
    profile_volume_multipliers: dict[str, VolumeMultiplier] = Field(
        default_factory=dict
    )
    apn_weights: dict[str, float] = Field(default_factory=dict)
    qos_distribution: list[QoSEntry] = Field(default_factory=list)
    partial_records: PartialRecordsConfig = Field(default_factory=PartialRecordsConfig)
    termination_causes: list[DataTerminationCause] = Field(default_factory=list)


class SmsEventConfig(BaseModel):
    delivery_success_rate: float = Field(default=0.97, ge=0.0, le=1.0)
    delivery_delay: DistributionWithBounds
    failure_causes: list[DataTerminationCause] = Field(default_factory=list)


class ConcurrencyConfig(BaseModel):
    max_voice: int | None = 1
    max_data: int | None = 1
    max_sms: int | None = None


class EventsConfig(BaseModel):
    voice: VoiceEventConfig
    data: DataEventConfig
    sms: SmsEventConfig
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------


class DuplicateRecordsConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.005, ge=0.0, le=1.0)
    max_time_shift_ms: int = 500


class MissingFieldsConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.01, ge=0.0, le=1.0)
    target_fields: list[str] = Field(default_factory=list)


class OrphanedRecordsConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.02, ge=0.0, le=1.0)


class TimestampAnomaliesConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.003, ge=0.0, le=1.0)
    types: list[str] = Field(default_factory=list)


class CorruptValuesConfig(BaseModel):
    enabled: bool = True
    rate: float = Field(default=0.002, ge=0.0, le=1.0)
    types: list[str] = Field(default_factory=list)


class AnomaliesConfig(BaseModel):
    enabled: bool = True
    duplicate_records: DuplicateRecordsConfig = Field(
        default_factory=DuplicateRecordsConfig
    )
    missing_fields: MissingFieldsConfig = Field(default_factory=MissingFieldsConfig)
    orphaned_records: OrphanedRecordsConfig = Field(
        default_factory=OrphanedRecordsConfig
    )
    timestamp_anomalies: TimestampAnomaliesConfig = Field(
        default_factory=TimestampAnomaliesConfig
    )
    corrupt_values: CorruptValuesConfig = Field(default_factory=CorruptValuesConfig)


# ---------------------------------------------------------------------------
# special_events
# ---------------------------------------------------------------------------


class Recurrence(BaseModel):
    days: list[str] = Field(default_factory=list)
    hours: list[int] = Field(default_factory=list)


class SpecialEventEffect(BaseModel):
    sms_rate_multiplier: float | None = None
    voice_rate_multiplier: float | None = None
    data_rate_multiplier: float | None = None
    voice_failure_rate_override: float | None = Field(default=None, ge=0.0, le=1.0)
    congestion_cells: list[int] = Field(default_factory=list)
    disabled_cells: list[int] = Field(default_factory=list)
    overflow_cells: list[int] = Field(default_factory=list)


class SpecialEventConfig(BaseModel):
    name: str
    time_range: TimeRange | None = None
    recurrence: Recurrence | None = None
    effect: SpecialEventEffect


# ---------------------------------------------------------------------------
# vendor_extensions
# ---------------------------------------------------------------------------


class VendorFieldConfig(BaseModel):
    key: str
    value: Distribution | str | dict[str, Any]


class VendorExtensionConfig(BaseModel):
    fields: list[VendorFieldConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class CDRGeneratorConfig(BaseModel):
    """Root configuration model — mirrors cdr_generator_config.yaml."""

    meta: MetaConfig
    network: NetworkConfig
    subscribers: SubscribersConfig
    events: EventsConfig
    anomalies: AnomaliesConfig = Field(default_factory=AnomaliesConfig)
    special_events: list[SpecialEventConfig] = Field(default_factory=list)
    vendor_extensions: dict[str, VendorExtensionConfig] = Field(default_factory=dict)
    interactive_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("special_events", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        return v if v is not None else []

    @field_validator("interactive_overrides", "vendor_extensions", mode="before")
    @classmethod
    def _coerce_none_to_dict(cls, v: Any) -> Any:
        return v if v is not None else {}
