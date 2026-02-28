"""SMS CDR generator -- MO/MT SMS pairs with delivery success/failure."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.generators.voice import _sample_distribution, _weighted_choice
from cdr_generator.models.cdr import CDRRecord


def generate_sms_cdr(
    sender: Subscriber,
    recipient: Subscriber,
    event_time: datetime,
    cell: Cell,
    smsc: NetworkElement,
    sms_cfg: dict,
    rng: np.random.Generator,
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

    Returns
    -------
    list[CDRRecord]
        One record (MO only) for failed delivery, two records (MO + MT) for
        successful delivery. MO and MT share a consolidation_id.
    """
    success_rate = sms_cfg.get("delivery_success_rate", 0.97)
    is_success = rng.random() < success_rate

    consolidation_id = uuid.UUID(
        bytes=bytes(rng.integers(0, 256, size=16, dtype="uint8"))
    ).hex

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
        cause_code = _pick_sms_failure_cause(sms_cfg, rng)
        mo.cause_for_termination = cause_code
        return [mo]

    # Successful delivery: add delivery delay for MT record
    delay = _sample_delivery_delay(sms_cfg, rng)
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


def _sample_delivery_delay(sms_cfg: dict, rng: np.random.Generator) -> float:
    """Sample SMS delivery delay from the configured distribution."""
    delay_cfg = sms_cfg.get("delivery_delay", {})
    dist = delay_cfg.get("distribution", {"type": "constant", "params": {"value": 1.0}})

    raw = _sample_distribution(dist, rng)

    min_s = delay_cfg.get("min_seconds", 0.5)
    max_s = delay_cfg.get("max_seconds", 86400)
    return float(max(min_s, min(raw, max_s)))


def _pick_sms_failure_cause(sms_cfg: dict, rng: np.random.Generator) -> int:
    """Pick an SMS failure cause from weighted list."""
    causes = sms_cfg.get("failure_causes", [])
    if not causes:
        return 1  # default generic failure

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng)
    # SMS failure causes typically don't have numeric codes in config,
    # use a default absent_subscriber code
    return 1
