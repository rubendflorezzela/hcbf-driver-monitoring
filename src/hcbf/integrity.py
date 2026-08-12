from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        checksum, relative = line.split(maxsplit=1)
        entries[relative.strip()] = checksum
    return entries


def verify_checksum_manifest(root: Path, manifest: Path) -> list[str]:
    problems: list[str] = []
    for relative, expected in read_checksum_manifest(manifest).items():
        target = root / relative
        if not target.is_file():
            problems.append(f"missing: {relative}")
            continue
        observed = sha256_file(target)
        if observed != expected:
            problems.append(f"hash mismatch: {relative}")
    return problems
