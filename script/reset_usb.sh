#!/usr/bin/env bash
set -e

VENDOR="38fb"
PRODUCT="1002"

echo "🔍 Searching for Reachy internal USB hub ($VENDOR:$PRODUCT)..."

HUB=""

for dev in /sys/bus/usb/devices/*; do
  if [[ -f "$dev/idVendor" && -f "$dev/idProduct" ]]; then
    v=$(cat "$dev/idVendor")
    p=$(cat "$dev/idProduct")

    if [[ "$v" == "$VENDOR" && "$p" == "$PRODUCT" ]]; then
      HUB=$(basename "$dev")
      break
    fi
  fi
done

if [[ -z "$HUB" ]]; then
  echo "❌ Reachy USB hub not found."
  echo "👉 Is Reachy plugged in?"
  exit 1
fi

echo "🔌 Found Reachy hub at: $HUB"
echo "🔁 Resetting hub..."

echo "$HUB" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null
sleep 2
echo "$HUB" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null

echo "✅ Reachy USB hub reset complete."
