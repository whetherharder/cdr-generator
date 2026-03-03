"""Pre-filled random number buffers for high-throughput CDR generation.

Instead of calling rng.lognormal(), rng.choice(), etc. once per CDR event,
_RngBuffer pre-fills arrays of _BATCH values and serves them from the
pre-filled array.  This amortises per-call Python overhead across _BATCH
events, replacing "Python dispatch + numpy scalar op" with "array index".
"""

from __future__ import annotations

import numpy as np

_BATCH: int = 2048


class _Buf:
    """Slot-optimised (array, position) pair for one RNG distribution.

    Using a single dict lookup (key → _Buf) plus attribute access is
    ~2x cheaper than the previous dual-dict design (one lookup for the
    array, one for the position), eliminating ~17 k dict operations
    across a typical 3 k-record generation run.
    """

    __slots__ = ("arr", "pos")

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.pos = 0


class _RngBuffer:
    """Batched random number source backed by a numpy Generator.

    All methods are semantically equivalent to direct rng calls but draw
    from pre-filled batches for lower per-event overhead.

    Usage::

        buf = _RngBuffer(np_rng)
        uid = buf.get_uuid()               # 32-char hex string
        x   = buf.get_float()              # uniform [0, 1)
        dur = buf.get_lognormal(4.0, 1.2)  # lognormal sample
        n   = buf.get_int(0, 1000)         # integer in [0, 1000)
        idx = buf.get_choice(3, (0.5, 0.3, 0.2))  # weighted choice
    """

    __slots__ = (
        "_rng",
        "_uuid_raw",
        "_uuid_pos",
        "_floats",
        "_float_pos",
        "_lognormal",
        "_ints",
        "_choice",
    )

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._uuid_raw: bytes = b""
        self._uuid_pos: int = _BATCH  # forces refill on first call
        self._floats: np.ndarray = np.empty(0)
        self._float_pos: int = _BATCH
        self._lognormal: dict[tuple[float, float], _Buf] = {}
        self._ints: dict[tuple[int, int], _Buf] = {}
        self._choice: dict[tuple, _Buf] = {}

    def get_uuid(self) -> str:
        """Return a 32-char hex UUID string from a pre-filled batch."""
        if self._uuid_pos >= _BATCH:
            self._uuid_raw = self._rng.bytes(16 * _BATCH)
            self._uuid_pos = 0
        pos = self._uuid_pos
        self._uuid_pos = pos + 1
        return self._uuid_raw[pos * 16 : pos * 16 + 16].hex()

    def get_float(self) -> float:
        """Return next pre-generated uniform [0, 1) float."""
        if self._float_pos >= _BATCH:
            self._floats = self._rng.random(_BATCH)
            self._float_pos = 0
        pos = self._float_pos
        self._float_pos = pos + 1
        return float(self._floats[pos])

    def get_lognormal(self, mu: float, sigma: float) -> float:
        """Return next pre-generated lognormal sample for (mu, sigma)."""
        key = (mu, sigma)
        b = self._lognormal.get(key)
        if b is None or b.pos >= _BATCH:
            b = _Buf(self._rng.lognormal(mu, sigma, size=_BATCH))
            self._lognormal[key] = b
        pos = b.pos
        b.pos = pos + 1
        return float(b.arr[pos])

    def get_int(self, lo: int, hi_excl: int) -> int:
        """Return next pre-generated integer in [lo, hi_excl)."""
        key = (lo, hi_excl)
        b = self._ints.get(key)
        if b is None or b.pos >= _BATCH:
            b = _Buf(self._rng.integers(lo, hi_excl, size=_BATCH))
            self._ints[key] = b
        pos = b.pos
        b.pos = pos + 1
        return int(b.arr[pos])

    def get_choice(self, n: int, p: tuple[float, ...]) -> int:
        """Return next pre-generated weighted choice from n items."""
        key = (n, p)
        b = self._choice.get(key)
        if b is None or b.pos >= _BATCH:
            b = _Buf(self._rng.choice(n, p=list(p), size=_BATCH))
            self._choice[key] = b
        pos = b.pos
        b.pos = pos + 1
        return int(b.arr[pos])
