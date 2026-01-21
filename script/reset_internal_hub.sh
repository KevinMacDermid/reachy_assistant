#!/usr/bin/env bash
# Resets the internal hub on Frisket
set -e

# IDs for the Terminus Technology internal hub
VENDOR="1a40"
PRODUCT="0101"

echo "🔍 Searching for Terminus internal USB hub ($VENDOR:$PRODUCT)..."

HUB_PATH=""

for dev in /sys/bus/usb/devices/*; do
  if [[ -f "$dev/idVendor" && -f "$dev/idProduct" ]]; then
    v=$(cat "$dev/idVendor")
    p=$(cat "$dev/idProduct")

    if [[ "$v" == "$VENDOR" && "$p" == "$PRODUCT" ]]; then
      # Ensure we get the specific hub controlling your Reachy (3-3)
      HUB_PATH=$(basename "$dev")
      echo "🔌 Found Terminus hub at: $HUB_PATH"

      echo "🔁 Resetting hub at $HUB_PATH..."
      # Unbinding/binding the hub resets all devices connected to it
      echo "$HUB_PATH" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null
      sleep 2
      echo "$HUB_PATH" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null
      echo "✅ Hub reset complete."
    fi
  fi
done

if [[ -z "$HUB_PATH" ]]; then
  echo "❌ Terminus internal hub not found."
  exit 1
fi
