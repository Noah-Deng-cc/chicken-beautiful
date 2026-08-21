#!/usr/bin/env bash
# 树莓派部署脚本：检查 Zero 2 W 前置条件并幂等安装服务；不保存任何凭据。
set -eu

APP_USER="${APP_USER:-dormassistant}"
APP_DIR="${APP_DIR:-/opt/dorm-assistant}"
ENV_DIR="${ENV_DIR:-/etc/dorm-assistant}"
CHECK_ONLY=false
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SOURCE_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: sudo deploy/install_pi.sh [--check-only]
  --check-only  仅验证 64 位 Raspberry Pi OS、Python、内存和硬件接口，不修改系统。
EOF
}

failures=0
check_required() {
  if "$@"; then printf 'PASS: %s\n' "$*"; else printf 'FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); fi
}
check_optional() {
  if "$@"; then printf 'PASS: %s\n' "$*"; else printf 'WARN: %s\n' "$*" >&2; fi
}
has_camera() { test -e /dev/video0 || command -v rpicam-hello >/dev/null 2>&1 || command -v libcamera-hello >/dev/null 2>&1; }
has_i2c() { compgen -G '/dev/i2c-*' >/dev/null; }
has_uart() { test -e /dev/serial0 || compgen -G '/dev/ttyAMA*' >/dev/null || compgen -G '/dev/ttyS*' >/dev/null; }
has_audio() { command -v aplay >/dev/null 2>&1 && aplay -l >/dev/null 2>&1; }
has_memory() {
  local memory_mb
  memory_mb="$(awk '/MemTotal/ { print int($2 / 1024) }' /proc/meminfo)"
  test "$memory_mb" -ge 450 || {
    # Zero 2 W firmware reserves enough RAM that Debian reports about 415 MB.
    test "$memory_mb" -ge 400 && test -r /proc/device-tree/model && grep -qai 'Raspberry Pi Zero 2 W' /proc/device-tree/model
  }
}
has_python() { python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; }
has_pios() {
  test -r /etc/os-release && {
    grep -qi 'raspbian\|raspberry pi' /etc/os-release || {
      test -r /proc/device-tree/model && grep -qai 'Raspberry Pi' /proc/device-tree/model
    }
  }
}

while test "$#" -gt 0; do
  case "$1" in
    --check-only) CHECK_ONLY=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

check_required test "$(uname -m)" = aarch64
check_required has_pios
check_required has_python
check_required has_memory
check_optional has_camera
check_optional has_i2c
check_optional has_uart
check_optional has_audio
if test "$failures" -ne 0 || test "$CHECK_ONLY" = true; then exit "$failures"; fi

if test "$(id -u)" -ne 0; then printf 'Run installation as root; service itself runs as %s.\n' "$APP_USER" >&2; exit 1; fi
if ! id "$APP_USER" >/dev/null 2>&1; then useradd --system --create-home --home-dir /var/lib/dorm-assistant --shell /usr/sbin/nologin "$APP_USER"; fi
for group in video audio i2c dialout; do getent group "$group" >/dev/null 2>&1 && usermod -a -G "$group" "$APP_USER"; done
install -d -m 0755 "$APP_DIR" "$ENV_DIR" /var/lib/dorm-assistant
cp -a "$SOURCE_DIR/." "$APP_DIR/"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip
"$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
if test ! -f "$APP_DIR/config/settings.pi.yaml"; then cp "$APP_DIR/config/settings.example.yaml" "$APP_DIR/config/settings.pi.yaml"; fi
if test ! -f "$ENV_DIR/dorm-assistant.env"; then cp "$APP_DIR/deploy/dorm-assistant.env.example" "$ENV_DIR/dorm-assistant.env"; fi
install -m 0644 "$APP_DIR/deploy/dorm-assistant.service" /etc/systemd/system/dorm-assistant.service
chown -R "$APP_USER:$APP_USER" "$APP_DIR" /var/lib/dorm-assistant
chown root:"$APP_USER" "$ENV_DIR/dorm-assistant.env"
chmod 0640 "$ENV_DIR/dorm-assistant.env"
systemctl daemon-reload
systemctl enable --now dorm-assistant.service
printf 'Installed. Edit %s and %s before enabling real hardware.\n' "$APP_DIR/config/settings.pi.yaml" "$ENV_DIR/dorm-assistant.env"
