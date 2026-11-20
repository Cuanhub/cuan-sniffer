#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server for CuanSniffer.
# Run as root:  bash setup.sh
#
# It installs and configures, and STARTS NOTHING. The jobs are enabled only at
# cutover (README.md, step 4), after the Mac's jobs are stopped, because two
# copies running at once would trade the same account twice.
set -euo pipefail

REPO_URL="https://github.com/Cuanhub/cuan-sniffer.git"
BRANCH="simplify-execution-core"
USER_NAME="cuan"
HOME_DIR="/home/${USER_NAME}"
APP="${HOME_DIR}/CuanSniffer"
PY_VERSION="3.13.5"        # the Mac's interpreter

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

timedatectl set-timezone UTC
apt-get update -y
apt-get install -y git curl ca-certificates unattended-upgrades ufw
dpkg-reconfigure -f noninteractive unattended-upgrades

# Firewall: SSH in, nothing else. The bot only makes outgoing requests.
ufw allow OpenSSH
ufw --force enable

id "${USER_NAME}" >/dev/null 2>&1 || adduser --disabled-password --gecos "" "${USER_NAME}"
install -d -o "${USER_NAME}" -g "${USER_NAME}" -m 700 "${HOME_DIR}/logs"
# Let the same SSH key that reaches root reach the bot's user, for copying files.
install -d -o "${USER_NAME}" -g "${USER_NAME}" -m 700 "${HOME_DIR}/.ssh"
[ -f /root/.ssh/authorized_keys ] && install -o "${USER_NAME}" -g "${USER_NAME}" -m 600 \
    /root/.ssh/authorized_keys "${HOME_DIR}/.ssh/authorized_keys"

sudo -u "${USER_NAME}" -H bash -s <<EOF
set -euo pipefail
cd "${HOME_DIR}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME_DIR}/.local/bin:\$PATH"
[ -d "${APP}" ] || git clone --branch "${BRANCH}" "${REPO_URL}" "${APP}"
cd "${APP}"
uv python install "${PY_VERSION}"
uv venv --python "${PY_VERSION}" .venv
uv pip install --python .venv/bin/python -r deploy/server/requirements.lock
mkdir -p research/levels/data research/trend/data
chmod 700 research/levels/data research/trend/data
EOF

install -m 644 "${APP}"/deploy/server/units/cuan-*.service "${APP}"/deploy/server/units/cuan-*.timer /etc/systemd/system/
systemctl daemon-reload

echo
echo "Installed. Nothing is running yet."
echo "Next: copy .env and the two data folders from the Mac (README.md, step 3)."
