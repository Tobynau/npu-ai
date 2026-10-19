#!/usr/bin/env bash
# Remove the DKMS amdxdna-usage module and go back to the stock in-tree driver.
set -euo pipefail
[[ $EUID -eq 0 ]] || exec sudo -- "$0" "$@"

dkms remove amdxdna-usage/1.0 --all || true
rm -rf /usr/src/amdxdna-usage-1.0
modprobe -r amdxdna && modprobe amdxdna
echo "Now using: $(modinfo -F filename amdxdna)"
