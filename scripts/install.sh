#!/bin/sh
set -eu

# Bootstrap only: platform selection, download, and argument forwarding belong
# to the signed installer binary.  GitHub HTTPS is the initial trust anchor.
case "$(uname -s):$(uname -m)" in
  Linux:x86_64|Linux:amd64) asset="ledgermind-installer-linux-x86_64" ;;
  Linux:aarch64|Linux:arm64) asset="ledgermind-installer-linux-aarch64" ;;
  *)
    printf '%s\n' "LedgerMind supports Linux x86_64 and Linux aarch64 only" >&2
    exit 3
    ;;
esac

base_url=${LEDGERMIND_RELEASE_BASE_URL:-https://github.com/sl4m3/ledgermind/releases/latest/download}
destination=${TMPDIR:-/tmp}/"$asset.$$"
cleanup() { rm -f "$destination"; }
trap cleanup EXIT INT TERM

printf '%s\n' "LedgerMind: downloading $asset" >&2
if command -v curl >/dev/null 2>&1; then
  if [ -t 2 ]; then
    curl --fail --location --show-error --progress-bar \
      "$base_url/$asset" -o "$destination"
  else
    curl -fsSL "$base_url/$asset" -o "$destination"
  fi
elif command -v wget >/dev/null 2>&1; then
  if [ -t 2 ]; then
    wget -O "$destination" "$base_url/$asset"
  else
    wget -qO "$destination" "$base_url/$asset"
  fi
else
  printf '%s\n' "curl or wget is required" >&2
  exit 5
fi
printf '%s\n' "LedgerMind: download complete; starting setup" >&2

chmod 700 "$destination"
export LEDGERMIND_RELEASE_BASE_URL="$base_url"

# When this bootstrap is streamed through `curl | sh`, stdin belongs to the
# shell reading the downloaded script rather than to the interactive wizard.
# Route interactive input to the user's terminal while leaving stdin untouched
# for explicitly non-interactive installs (for example token_stdin).
interactive=1
for arg in "$@"; do
  if [ "$arg" = "--non-interactive" ]; then
    interactive=0
    break
  fi
done

if [ "$interactive" -eq 1 ] && [ ! -t 0 ]; then
  if [ -r /dev/tty ]; then
    exec "$destination" "$@" </dev/tty
  fi
  printf '%s\n' "an interactive terminal is required; use --non-interactive with --config for piped installs" >&2
  exit 4
fi

exec "$destination" "$@"
