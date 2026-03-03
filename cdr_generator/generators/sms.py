"""SMS CDR generator -- MO/MT SMS pairs with delivery success/failure."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.generators.voice import _sample_distribution, _weighted_choice
from cdr_generator.models.cdr import CDRRecord

if TYPE_CHECKING:
    from cdr_generator.engine.rng_buffer import _RngBuffer


def generate_sms_cdr(
    sender: Subscriber,
    recipient: Subscriber,
    event_time: datetime,
    cell: Cell,
    smsc: NetworkElement,
    sms_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> list[CDRRecord]:
    """Generate SMS CDR records for a single message.

    Parameters
    ----------
    sender:
        A-party subscriber sending the SMS.
    recipient:
        B-party subscriber receiving the SMS.
    event_time:
        Timestamp of the SMS submission.
    cell:
        Cell where the sender is located.
    smsc:
        SMSC network element handling the message.
    sms_cfg:
        SMS event configuration dict (from config.events.sms).
    rng:
        Numpy random generator for deterministic sampling.
    _buf:
        Optional pre-filled RNG buffer for high-throughput generation.

    Returns
    -------
    list[CDRRecord]
        One record (MO only) for failed delivery, two records (MO + MT) for
        successful delivery. MO and MT share a consolidation_id.
    """
    success_rate = sms_cfg.get("delivery_success_rate", 0.97)
    roll = _buf.get_float() if _buf is not None else float(rng.random())
    is_success = roll < success_rate

    if _buf is not None:
        consolidation_id = _buf.get_uuid()
    else:
        consolidation_id = bytes(rng.integers(0, 256, size=16, dtype="uint8")).hex()

    mo = CDRRecord(
        record_type="mo_sms",
        served_imsi=sender.imsi,
        served_msisdn=sender.msisdn,
        served_imei=sender.imei,
        event_timestamp=event_time,
        calling_number=sender.msisdn,
        called_number=recipient.msisdn,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=smsc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    if not is_success:
        cause_code = _pick_sms_failure_cause(sms_cfg, rng, _buf=_buf)
        mo.cause_for_termination = cause_code
        return [mo]

    # Successful delivery: add delivery delay for MT record
    delay = _sample_delivery_delay(sms_cfg, rng, _buf=_buf)
    mt_time = event_time + timedelta(seconds=delay)

    mt = CDRRecord(
        record_type="mt_sms",
        served_imsi=recipient.imsi,
        served_msisdn=recipient.msisdn,
        served_imei=recipient.imei,
        event_timestamp=mt_time,
        calling_number=sender.msisdn,
        called_number=recipient.msisdn,
        first_cell_id=recipient.home_cell_id,
        last_cell_id=recipient.home_cell_id,
        serving_ne_id=smsc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    return [mo, mt]


def _sample_delivery_delay(
    sms_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> float:
    """Sample SMS delivery delay from the configured distribution."""
    delay_cfg = sms_cfg.get("delivery_delay", {})
    dist = delay_cfg.get("distribution", {"type": "constant", "params": {"value": 1.0}})

    raw = _sample_distribution(dist, rng, _buf=_buf)

    min_s = delay_cfg.get("min_seconds", 0.5)
    max_s = delay_cfg.get("max_seconds", 86400)
    return float(max(min_s, min(raw, max_s)))


def _pick_sms_failure_cause(
    sms_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Pick an SMS failure cause from weighted list."""
    causes = sms_cfg.get("failure_causes", [])
    if not causes:
        return 1  # default generic failure

    weights = [c["weight"] for c in causes]
    _weighted_choice(weights, rng, _buf=_buf)  # consume RNG for determinism
    # SMS failure causes typically don't have numeric codes in config,
    # use a default absent_subscriber code
    return 1
