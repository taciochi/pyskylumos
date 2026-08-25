"""Validate release artifact size, contents, metadata, and package version."""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from pathlib import Path

from packaging.specifiers import SpecifierSet

from pyskylumos._version import __version__

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = (".pdf", ".zip")


def members(path: Path) -> list[str]:
    """Return normalized archive member names."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def metadata(path: Path) -> email.message.Message:
    """Return parsed wheel METADATA or sdist PKG-INFO."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.endswith("/METADATA"))
            return email.message_from_bytes(archive.read(member))
    with tarfile.open(path) as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("/PKG-INFO"))
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"Could not read metadata from {path}.")
        return email.message_from_binary_file(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("distribution_directory", type=Path)
    args = parser.parse_args()
    artifacts = sorted(args.distribution_directory.glob(f"pyskylumos-{__version__}*"))
    if len(artifacts) != 2:
        raise SystemExit(f"Expected one wheel and one sdist, found: {artifacts}")

    problems: list[str] = []
    for artifact in artifacts:
        if artifact.stat().st_size > MAX_ARTIFACT_BYTES:
            problems.append(f"Artifact exceeds 10 MiB: {artifact}")
        forbidden = [
            name for name in members(artifact) if name.lower().endswith(FORBIDDEN_SUFFIXES)
        ]
        if forbidden:
            problems.append(f"Forbidden PDFs/ZIPs in {artifact.name}: {forbidden}")

        artifact_metadata = metadata(artifact)
        if artifact_metadata["Version"] != __version__:
            problems.append(
                f"Version mismatch in {artifact.name}: {artifact_metadata['Version']} != {__version__}"
            )
        if SpecifierSet(artifact_metadata["Requires-Python"]) != SpecifierSet(">=3.12,<3.15"):
            problems.append(
                f"Unexpected Requires-Python in {artifact.name}: "
                f"{artifact_metadata['Requires-Python']}"
            )
        if "jpl" not in artifact_metadata.get_all("Provides-Extra", []):
            problems.append(f"Missing Provides-Extra: jpl in {artifact.name}")
        requirements = artifact_metadata.get_all("Requires-Dist", [])
        if not any("jplephem<3,>=2.17" in requirement for requirement in requirements):
            problems.append(f"Missing bounded jplephem requirement in {artifact.name}")

    if problems:
        raise SystemExit("\n".join(problems))

    print("distribution manifest: PASS")


if __name__ == "__main__":
    main()
