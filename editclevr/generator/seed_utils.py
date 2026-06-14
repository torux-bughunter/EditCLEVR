from __future__ import annotations

import hashlib


def derive_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    payload = "::".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % modulo
