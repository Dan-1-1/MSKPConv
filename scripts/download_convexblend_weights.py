"""Download and verify the complete final ConvexBlend weight bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "models" / "WEIGHTS_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    asset = manifest["release_asset"]
    destination = REPOSITORY_ROOT / "models" / "convexblend" / "final"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mskpconv_weights_") as temp_dir:
        archive = Path(temp_dir) / asset["name"]
        print(f"Downloading {asset['url']}")
        with urllib.request.urlopen(asset["url"]) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        actual = sha256(archive)
        if actual != asset["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {actual} != {asset['sha256']}")
        safe_extract(archive, destination)

    for item in manifest["convexblend_files"]:
        path = REPOSITORY_ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Weight verification failed: {path}")
    print("ConvexBlend weights downloaded and verified.")


if __name__ == "__main__":
    main()
