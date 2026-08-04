from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ledgermind_local.cli import main
from ledgermind_local.config import LocalConfig
from ledgermind_local.core_gateway.doctor import build_core_doctor_report
from ledgermind_local.core_gateway.signing import calculate_sha256
from ledgermind_local.paths import ServicePaths


def _write_signed_fake_core(paths: ServicePaths) -> None:
    binary = paths.core_data_dir / "bin" / "fake-core"
    signature = paths.core_data_dir / "bin" / "fake-core.sig"
    public_key = paths.core_data_dir / "bin" / "fake-core.pub"
    binary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    script = f"""#!{sys.executable}
import json
import sys


def read_frame():
    header = sys.stdin.buffer.read(4)
    if not header:
        return None
    size = int.from_bytes(header, 'big')
    return json.loads(sys.stdin.buffer.read(size))


def send(request_id, result):
    payload = json.dumps({{
        'protocol_version': 1,
        'request_id': request_id,
        'status': 'ok',
        'result': result,
    }}, separators=(',', ':')).encode()
    sys.stdout.buffer.write(len(payload).to_bytes(4, 'big') + payload)
    sys.stdout.buffer.flush()


while True:
    request = read_frame()
    if request is None:
        break
    operation = request['operation']
    request_id = request['request_id']
    if operation == 'handshake':
        send(request_id, {{
            'protocol_version': 1,
            'server_name': 'signed-fake-core',
            'version': 'test-core-1',
            'operations': ['handshake', 'health', 'shutdown'],
        }})
    elif operation == 'health':
        send(request_id, {{
            'healthy': True,
            'backend': 'rust',
            'protocol_version': 1,
            'schema_version': 2,
            'environment_keys': [
                'LANG', 'LC_ALL', 'RUST_BACKTRACE',
                'LEDGERMIND_CORE_DATA_DIR', 'PWD'
            ],
        }})
    elif operation == 'shutdown':
        send(request_id, {{'stopped': True}})
        break
"""
    binary.write_text(script, encoding="utf-8")
    os.chmod(binary, 0o700)
    private_key = Ed25519PrivateKey.generate()
    signature.write_bytes(private_key.sign(binary.read_bytes()))
    public_key.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )


def _doctor_config() -> LocalConfig:
    return LocalConfig(
        config_version=1,
        core_backend="process",
        core_binary_path="bin/fake-core",
        core_signature_path="bin/fake-core.sig",
        core_public_key_path="bin/fake-core.pub",
        verify_core_signature=True,
        require_core_network_isolation=False,
    )


def test_core_doctor_reports_missing_signature_without_launching(tmp_path: Path) -> None:
    paths = ServicePaths(tmp_path / "local")
    paths.core_data_dir.mkdir(mode=0o700, parents=True)
    binary = paths.core_data_dir / "bin" / "fake-core"
    binary.parent.mkdir(mode=0o700, exist_ok=True)
    binary.write_bytes(b"unsigned core")

    report = build_core_doctor_report(paths=paths, config=_doctor_config())

    assert report["ok"] is False
    assert report["signature"]["status"] == "failed"
    assert report["health"]["detail"] == "Core was not launched"
    assert report["environment"]["values_exposed"] is False
    assert report["sandbox"]["capabilities"]["binary_signature_verified"] is False


def test_core_doctor_cli_verifies_binary_and_reports_runtime_without_secrets(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "local"
    paths = ServicePaths(home)
    paths.home.mkdir(mode=0o700, parents=True)
    _write_signed_fake_core(paths)
    config = _doctor_config()
    paths.config_file.write_text(config.to_json(), encoding="utf-8")
    monkeypatch.setenv("LEDGERMIND_API_KEY", "[REDACTED]")

    code = main(["--home", str(home), "core", "doctor", "--json"])

    assert code == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["ok"] is True
    assert report["binary"]["sha256"] == calculate_sha256(paths.core_data_dir / "bin" / "fake-core")
    assert report["signature"]["status"] == "verified"
    assert report["version"] == "test-core-1"
    assert report["server_name"] == "signed-fake-core"
    assert report["health"]["protocol_version"] == 1
    assert report["health"]["schema_version"] == 2
    assert report["sandbox"]["level"] in {"full", "partial"}
    assert report["sandbox"]["capabilities"]["binary_signature_verified"] is True
    assert "missing_requirements" in report["sandbox"]
    assert report["environment"]["secret_like_keys"] == []
    assert report["environment"]["unexpected_keys"] == []
    assert report["environment"]["values_exposed"] is False
    assert "[REDACTED]" not in output
    assert "LEDGERMIND_API_KEY" not in output
