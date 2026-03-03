"""Runtime asset models for cells, network elements, and subscribers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Cell(BaseModel):
    """A single cell in the network topology."""

    cell_id: int
    tac: int
    ecgi: str
    lat: float
    lon: float
    azimuth: int = Field(ge=0, lt=360)
    sector: int = Field(ge=1)
    cell_type: str
    capacity: str
    neighbors: list[int] = Field(default_factory=list)


class NetworkElement(BaseModel):
    """A network element (MSC, SGW, PGW, SMSC) serving a set of TACs."""

    id: str
    ne_type: str
    vendor: str
    address: str = ""
    serves_tacs: list[int] = Field(default_factory=list)
    serves_apns: list[str] = Field(default_factory=list)
    extensions_template: dict[str, object] = Field(default_factory=dict)
    produces: list[str] = Field(default_factory=list)


class Subscriber(BaseModel):
    """A generated subscriber with identity and network assignment."""

    imsi: str
    msisdn: str
    imei: str
    profile_name: str
    home_cell_id: int
    work_cell_id: int | None = None
    serving_ne_id: str


class Assets(BaseModel):
    """Bundled runtime assets: cells, network elements, and subscribers."""

    cells: list[Cell]
    network_elements: list[NetworkElement]
    subscribers: list[Subscriber]


class AssetManifest(BaseModel):
    """Manifest tracking asset generation provenance."""

    config_hash: str
    generated_at: datetime
    cell_count: int = Field(ge=0)
    ne_count: int = Field(ge=0)
    subscriber_count: int = Field(ge=0)
