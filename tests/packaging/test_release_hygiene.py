from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release-local.py"


class LocalReleaseScriptTests(unittest.TestCase):
    def _head(self) -> tuple[str, int]:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        timestamp = int(
            subprocess.run(
                ["git", "show", "-s", "--format=%ct", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, timestamp

    def _manifest(self, directory: Path, commit: str, commit_timestamp: int) -> Path:
        artifact = directory / "local-test-artifact.whl"
        artifact.write_bytes(b"release-test")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        with (ROOT / "pyproject.toml").open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
        manifest = directory / "ledgermind-local-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "format": "ledgermind-release-manifest-v1",
                    "component": "ledgermind-local",
                    "source_commit": commit,
                    "commit_sha": commit,
                    "source_commit_timestamp": commit_timestamp,
                    "version": version,
                    "build_time_epoch": commit_timestamp + 1,
                    "install_smoke": {"passed": True},
                    "artifacts": [
                        {"name": artifact.name, "sha256": digest, "size_bytes": artifact.stat().st_size}
                    ],
                    "sha256": {artifact.name: digest},
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_help_is_available_without_building(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dependency-wheel", result.stdout)

    def test_verify_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "verify", "--manifest", str(Path(temporary) / "missing.json")],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("manifest is required", result.stderr)

    def test_verify_rejects_manifest_from_another_commit(self) -> None:
        commit, timestamp = self._head()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._manifest(Path(temporary), "0" * len(commit), timestamp)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "verify", "--manifest", str(manifest)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("source commit does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
