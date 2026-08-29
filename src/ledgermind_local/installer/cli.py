"""Public ``ledgermind`` installer command surface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ledgermind_local.runtime.supervisor import RuntimeSupervisor

from .errors import ConfigurationError, ExitCode, InstallerError
from .models import InstallerConfig
from .non_interactive import load_non_interactive_config
from .operations.configure import configure
from .operations.doctor import doctor
from .operations.export_config import export_config
from .operations.import_config import import_config
from .operations.install import install, install_plan
from .operations.integrations import (
    connect_integration,
    disconnect_integration,
    discover_integrations,
    integration_status,
    set_integration_enabled,
)
from .operations.repair import repair
from .operations.status import status
from .operations.uninstall import uninstall
from .operations.update import update
from .paths import InstallerPaths
from .result import InstallResult, ResultStep
from .schema import schema
from .targets.registry import target_ids


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--home", type=Path, help="test/rootless home override")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgermind", description="LedgerMind universal installer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_subparsers = install_parser.add_subparsers(dest="install_command")
    schema_parser = install_subparsers.add_parser("schema")
    schema_parser.add_argument(
        "--name",
        choices=("install-config", "install-manifest", "install-result"),
        default="install-config",
    )
    schema_parser.add_argument("--json", action="store_true", dest="json_output")
    plan_parser = install_subparsers.add_parser("plan")
    _common(plan_parser)
    plan_parser.add_argument("--manifest", type=Path)
    for option in (install_parser,):
        _common(option)
        option.add_argument("--manifest", type=Path)
        option.add_argument("--bundle", type=Path)
        option.add_argument("--manifest-signature", type=Path)
        option.add_argument("--skip-provider-probe", action="store_true")

    configure_parser = subparsers.add_parser("configure")
    _common(configure_parser)
    configure_parser.add_argument("--validate-providers", action="store_true")

    doctor_parser = subparsers.add_parser("doctor")
    _common(doctor_parser)
    doctor_parser.add_argument("--no-smoke", action="store_true")

    repair_parser = subparsers.add_parser("repair")
    _common(repair_parser)

    update_parser = subparsers.add_parser("update")
    _common(update_parser)
    update_parser.add_argument("--manifest", type=Path, required=True)
    update_parser.add_argument("--bundle", type=Path, required=True)
    update_parser.add_argument("--skip-provider-probe", action="store_true")

    uninstall_parser = subparsers.add_parser("uninstall")
    _common(uninstall_parser)
    uninstall_parser.add_argument("--purge-data", action="store_true")
    uninstall_parser.add_argument("--purge-config", action="store_true")
    uninstall_parser.add_argument("--yes", action="store_true")

    status_parser = subparsers.add_parser("status")
    _common(status_parser)

    integrations_parser = subparsers.add_parser("integrations")
    integration_commands = integrations_parser.add_subparsers(
        dest="integration_command", required=True
    )
    integrations_discover = integration_commands.add_parser("discover")
    _common(integrations_discover)
    integrations_status = integration_commands.add_parser("status")
    _common(integrations_status)
    integrations_connect = integration_commands.add_parser("connect")
    _common(integrations_connect)
    integrations_connect.add_argument("integration", choices=target_ids())
    integrations_connect.add_argument("--disabled", action="store_true")
    for action in ("enable", "disable", "disconnect"):
        action_parser = integration_commands.add_parser(action)
        _common(action_parser)
        action_parser.add_argument("integration", choices=target_ids())

    export_parser = subparsers.add_parser("export-config")
    _common(export_parser)
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.add_argument("--include-secrets", action="store_true")

    import_parser = subparsers.add_parser("import-config")
    _common(import_parser)
    import_parser.add_argument("--file", type=Path, required=True)
    import_parser.add_argument("--validate-providers", action="store_true")
    import_parser.add_argument("--register-target", action="store_true")
    import_parser.add_argument("--delete-config-after-import", action="store_true")

    runtime_parser = subparsers.add_parser("runtime")
    _common(runtime_parser)
    runtime_subparsers = runtime_parser.add_subparsers(
        dest="runtime_command", required=True
    )

    def _runtime_flags(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--json", action="store_true", dest="json_output", default=argparse.SUPPRESS
        )
        command_parser.add_argument(
            "--non-interactive", action="store_true", default=argparse.SUPPRESS
        )
        command_parser.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        command_parser.add_argument(
            "--dry-run", action="store_true", default=argparse.SUPPRESS
        )
        command_parser.add_argument(
            "--verbose", action="store_true", default=argparse.SUPPRESS
        )
        command_parser.add_argument("--home", type=Path, default=argparse.SUPPRESS)

    acquire = runtime_subparsers.add_parser("acquire")
    _runtime_flags(acquire)
    acquire.add_argument("--client", required=True)
    acquire.add_argument("--session-id", required=True)
    heartbeat = runtime_subparsers.add_parser("heartbeat")
    _runtime_flags(heartbeat)
    heartbeat.add_argument("--lease-id", required=True)
    release = runtime_subparsers.add_parser("release")
    _runtime_flags(release)
    release.add_argument("--lease-id", required=True)
    runtime_status = runtime_subparsers.add_parser("status")
    _runtime_flags(runtime_status)
    stop = runtime_subparsers.add_parser("stop")
    _runtime_flags(stop)
    stop.add_argument("--force", action="store_true")
    return parser


def _paths(args: argparse.Namespace) -> InstallerPaths:
    return InstallerPaths(home_override=getattr(args, "home", None))


def _runtime(paths: InstallerPaths) -> RuntimeSupervisor:
    config: InstallerConfig | None = None
    settings: dict[str, float] = {
        "idle_shutdown_seconds": 60.0,
        "lease_ttl_seconds": 30.0,
    }
    if paths.config_file.is_file():
        try:
            from .config_writer import load_installer_config

            config = load_installer_config(paths.config_file)
            settings.update(
                {
                    "idle_shutdown_seconds": config.runtime.idle_shutdown_seconds,
                    "lease_ttl_seconds": config.runtime.lease_ttl_seconds,
                }
            )
        except (OSError, ValueError):
            return RuntimeSupervisor(
                paths,
                idle_shutdown_seconds=settings["idle_shutdown_seconds"],
                lease_ttl_seconds=settings["lease_ttl_seconds"],
            )
    local = paths.current_link / "bin" / "ledgermind-local"
    commands: dict[str, Sequence[str]] = {}
    if local.is_file() and local.stat().st_mode & 0o111:
        commands["local"] = (
            str(local),
            "serve",
            "--home",
            str(paths.data_dir / "local"),
        )
    if (
        config is not None
        and config.embedding.mode == "local"
        and config.embedding.local is not None
    ):
        from .hardware import choose_device

        local_config = config.embedding.local
        device: str = local_config.device
        if device == "auto":
            device = choose_device("auto", supported={"cpu", "cuda", "rocm"}).kind
        model_path = local_config.model_path or local_config.model_storage_path
        if model_path is None:
            model_path = str(paths.models_dir / local_config.catalog_id)
        commands["embedding"] = (
            sys.executable,
            "-m",
            "ledgermind_local.installer.embeddings.serve",
            "--model-path",
            str(Path(model_path).expanduser()),
            "--device",
            device,
            "--threads",
            str(local_config.threads or 4),
            "--gpu-layers",
            "99" if device != "cpu" else "0",
            "--port",
            "8766",
        )
    return RuntimeSupervisor(
        paths,
        idle_shutdown_seconds=settings["idle_shutdown_seconds"],
        lease_ttl_seconds=settings["lease_ttl_seconds"],
        commands=commands,
    )


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Accept global options before the command as the legacy CLI did."""

    values = {"--home", "--config", "--manifest", "--bundle", "--manifest-signature"}
    raw = list(argv)
    if not raw or not raw[0].startswith("-"):
        return raw
    prefix: list[str] = []
    index = 0
    while index < len(raw) and raw[index].startswith("-"):
        option = raw[index]
        prefix.append(option)
        if option in values:
            index += 1
            if index >= len(raw):
                return raw
            prefix.append(raw[index])
        index += 1
    if index >= len(raw):
        return raw
    return [raw[index], *prefix, *raw[index + 1 :]]


def _config(args: argparse.Namespace) -> tuple[InstallerConfig, str | None, str | None]:
    if args.config is not None and args.non_interactive:
        return load_non_interactive_config(args.config)
    if args.config is not None:
        from .config_writer import load_installer_config

        try:
            return load_installer_config(args.config), None, None
        except (OSError, UnicodeError, ValueError) as exc:
            raise ConfigurationError("install config is invalid") from exc
    if args.non_interactive:
        raise InstallerError("--config is required in non-interactive mode")
    from .wizard import build_interactive_config

    return build_interactive_config(), None, None


def _result(
    operation: str,
    *,
    payload: dict[str, Any] | None = None,
    paths: InstallerPaths | None = None,
) -> InstallResult:
    result = InstallResult(operation)
    if paths is not None:
        result.paths.update(paths.as_dict())
    if payload:
        for key in ("current", "release_dir", "config", "out", "plugin_root"):
            if key not in payload:
                continue
            value = payload[key]
            if key == "config" and isinstance(value, dict):
                result.paths["config_metadata"] = value
            else:
                result.paths[key] = value
        if "profiles" in payload and isinstance(payload["profiles"], list):
            result.profiles = payload["profiles"]
        elif isinstance(payload.get("plan"), dict) and isinstance(
            payload["plan"].get("profiles"), list
        ):
            result.profiles = payload["plan"]["profiles"]
        if "runtime" in payload and isinstance(payload["runtime"], dict):
            result.runtime = payload["runtime"]
        elif operation.startswith("runtime "):
            result.runtime = dict(payload)
        elif isinstance(payload.get("plan"), dict) and isinstance(
            payload["plan"].get("runtime"), dict
        ):
            result.runtime = payload["plan"]["runtime"]
        if "smoke_test" in payload and isinstance(payload["smoke_test"], dict):
            result.smoke_test = payload["smoke_test"]
        payload_status = payload.get("status")
        step_status = (
            payload_status
            if isinstance(payload_status, str)
            and payload_status in {"passed", "success", "partial", "dry_run", "failed"}
            else "passed"
        )
        result.steps.append(ResultStep(operation, step_status, data=dict(payload)))
        if step_status == "dry_run":
            result.status = "dry_run"
        elif step_status == "failed":
            result.status = "failed"
        elif step_status == "partial":
            result.status = "partial"
            result.exit_code = ExitCode.ADAPTER_FAILED
            failures = payload.get("integration_failures", [])
            if failures:
                result.warning(
                    "platform installed; integrations failed: "
                    + ", ".join(str(item) for item in failures)
                )
    return result


def _emit(result: InstallResult, *, json_output: bool) -> int:
    if json_output:
        print(result.to_json())
    else:
        print(
            f"{result.operation}: {result.status} (exit_code={int(result.exit_code)})"
        )
        for step in result.steps:
            print(f"- {step.name}: {step.status}")
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for error in result.errors:
            print(f"error: {error}", file=sys.stderr)
    return int(result.exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = ["install"]
    args = parser.parse_args(_normalize_argv(raw_argv))
    operation = args.command
    try:
        paths = _paths(args)
        if operation == "install" and args.install_command == "schema":
            result = _result("install schema", payload=schema(args.name))
            return _emit(result, json_output=bool(args.json_output))
        if operation == "install" and args.install_command == "plan":
            config, _, _ = _config(args)
            manifest = None
            if args.manifest:
                from .manifest import load_manifest

                manifest = load_manifest(args.manifest)
            result = _result(
                "install plan",
                payload=install_plan(config, paths=paths, manifest=manifest),
                paths=paths,
            )
            return _emit(result, json_output=bool(args.json_output))
        if operation == "install":
            config, generation_stdin, embedding_stdin = _config(args)
            payload = install(
                config=config,
                paths=paths,
                manifest_path=args.manifest,
                bundle=args.bundle,
                manifest_signature=args.manifest_signature,
                dry_run=bool(args.dry_run),
                skip_provider_probe=bool(args.skip_provider_probe),
                generation_stdin=generation_stdin,
                embedding_stdin=embedding_stdin,
            )
            result = _result("install", payload=payload, paths=paths)
            return _emit(result, json_output=bool(args.json_output))
        if operation == "configure":
            config, generation_stdin, embedding_stdin = _config(args)
            payload = configure(
                config=config,
                paths=paths,
                validate_providers=bool(args.validate_providers),
                generation_stdin=generation_stdin,
                embedding_stdin=embedding_stdin,
                dry_run=bool(args.dry_run),
            )
            return _emit(
                _result("configure", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "doctor":
            payload = doctor(paths=paths, full_smoke=not bool(args.no_smoke))
            result = _result("doctor", payload=payload, paths=paths)
            if payload.get("status") != "passed":
                result.fail(ExitCode.DOCTOR_FAILED, "doctor found failures")
            return _emit(result, json_output=bool(args.json_output))
        if operation == "repair":
            payload = repair(paths=paths, dry_run=bool(args.dry_run))
            return _emit(
                _result("repair", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "update":
            config, generation_stdin, embedding_stdin = _config(args)
            payload = update(
                config=config,
                paths=paths,
                manifest_path=args.manifest,
                bundle=args.bundle,
                dry_run=bool(args.dry_run),
                skip_provider_probe=bool(args.skip_provider_probe),
                generation_stdin=generation_stdin,
                embedding_stdin=embedding_stdin,
            )
            return _emit(
                _result("update", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "uninstall":
            payload = uninstall(
                paths=paths,
                purge_data=bool(args.purge_data),
                purge_config=bool(args.purge_config),
                yes=bool(args.yes),
                dry_run=bool(args.dry_run),
            )
            return _emit(
                _result("uninstall", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "status":
            return _emit(
                _result("status", payload=status(paths=paths), paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "integrations":
            if args.integration_command == "discover":
                payload = discover_integrations()
            elif args.integration_command == "status":
                payload = integration_status(paths=paths)
            elif args.integration_command == "connect":
                payload = connect_integration(
                    paths=paths,
                    target_id=args.integration,
                    enabled=not bool(args.disabled),
                    dry_run=bool(args.dry_run),
                )
            elif args.integration_command == "disconnect":
                payload = disconnect_integration(
                    paths=paths,
                    target_id=args.integration,
                    dry_run=bool(args.dry_run),
                )
            else:
                payload = set_integration_enabled(
                    paths=paths,
                    target_id=args.integration,
                    enabled=args.integration_command == "enable",
                    dry_run=bool(args.dry_run),
                )
            return _emit(
                _result(
                    "integrations " + args.integration_command,
                    payload=payload,
                    paths=paths,
                ),
                json_output=bool(args.json_output),
            )
        if operation == "export-config":
            payload = export_config(
                paths=paths, out=args.out, include_secrets=bool(args.include_secrets)
            )
            return _emit(
                _result("export-config", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "import-config":
            payload = import_config(
                paths=paths,
                file=args.file,
                validate_providers=bool(args.validate_providers),
                register_target=bool(args.register_target),
                delete_after_import=bool(args.delete_config_after_import),
            )
            return _emit(
                _result("import-config", payload=payload, paths=paths),
                json_output=bool(args.json_output),
            )
        if operation == "runtime":
            supervisor = _runtime(paths)
            if args.runtime_command == "acquire":
                payload = supervisor.acquire(
                    client=args.client, session_id=args.session_id
                )
            elif args.runtime_command == "heartbeat":
                payload = supervisor.heartbeat(args.lease_id)
            elif args.runtime_command == "release":
                payload = supervisor.release(args.lease_id)
            elif args.runtime_command == "status":
                payload = supervisor.status()
            else:
                payload = supervisor.stop(force=bool(args.force))
            return _emit(
                _result(
                    "runtime " + args.runtime_command,
                    payload=payload,
                    paths=paths,
                ),
                json_output=bool(args.json_output),
            )
        parser.error("unsupported installer command")
    except InstallerError as exc:
        result = InstallResult(operation, exit_code=exc.exit_code)
        result.fail(exc.exit_code, str(exc))
        return _emit(result, json_output=bool(getattr(args, "json_output", False)))
    except PermissionError:
        result = InstallResult(operation, exit_code=ExitCode.PERMISSION_ERROR)
        result.fail(result.exit_code, "permission denied")
        return _emit(result, json_output=bool(getattr(args, "json_output", False)))
    except ValueError as exc:
        result = InstallResult(operation, exit_code=ExitCode.CONFIGURATION_INVALID)
        result.fail(result.exit_code, str(exc))
        return _emit(result, json_output=bool(getattr(args, "json_output", False)))
    except Exception as exc:  # noqa: BLE001
        result = InstallResult(operation, exit_code=ExitCode.TRANSACTION_FAILED)
        result.fail(result.exit_code, str(exc))
        return _emit(result, json_output=bool(getattr(args, "json_output", False)))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
