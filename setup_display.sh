#!/bin/bash
# setup_display.sh
# Forces the Raspberry Pi's primary Micro-HDMI output (HDMI-A-1) to 1920x1080 at 60Hz.
# IMPORTANT: Run `xrandr` first to identify your connected display's name. It might be HDMI-1, HDMI-A-1, etc.
# Adjust the --output parameter below to match your display's name.

echo "--- Applying 1080p@60Hz display mode ---"

# Get the display name (usually HDMI-1 or HDMI-A-1 on RPi 4/5)
# This is a common name, but verify with `xrandr`
DISPLAY_NAME="HDMI-A-1"

# Check if the mode already exists, if not, create it with `cvt`
MODELINE=$(cvt 1920 1080 60 | sed -n '2p' | sed 's/.*Modeline //')
MODE_NAME=$(echo "$MODELINE" | awk '{print $1}' | tr -d '"')

# Check if the mode is already added
if ! xrandr | grep -q "$MODE_NAME"; then
    echo "Mode $MODE_NAME not found. Creating and adding new mode."
    xrandr --new-mode $MODELINE
    xrandr --add-mode $DISPLAY_NAME $MODE_NAME
else
    echo "Mode $MODE_NAME already exists."
fi

# Set the desired mode
echo "Setting output $DISPLAY_NAME to $MODE_NAME"
xrandr --output $DISPLAY_NAME --mode $MODE_NAME --rate 60

echo "--- Display setup complete. ---"
echo "NOTE: For a permanent solution, you should edit /boot/config.txt with the following lines:"
echo "hdmi_group=2"
echo "hdmi_mode=82"
echo "This forces 1080p DMT mode on boot."
