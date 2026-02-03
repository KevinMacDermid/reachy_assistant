# Reachy Mini Assistant
This is a small project to set up the Reachy Mini robot as a voice assistant


## Wake Word
Using OpenWakeWord models, here's a Reachy specific one someone trained
https://huggingface.co/spaces/luisomoreau/hey_reachy_wake_word_detection


# Installation
## Install PortAudio
This is required for OpenWakeWord, and is not installed automatically.

```bash
sudo apt install portaudio19-dev
```

## Install Python-SDK
https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md

**That includes instructions for allowing access to the USB**

Had and issue with Direction of Arrival with USB permissions, 
had to update /etc/udev/rules.d/99-reachy-mini.rules to
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"
# Target the raw USB hardware (usb) - This fixes the USBError Errno 13
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
# Do the same for the Audio/Camera if they are causing issues
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", MODE="0666", GROUP="dialout"
```
The last line was the one that seemed to fix it.

## Install Dependencies
Run 
```bash
uv sync
```

## Add Run Internal to Sudoers
On Frisket, had issue where the internal hub of the laptop gets stuck, added a script for this 
`./script/reset_internal_hub.sh`, had to adjust sudoers to allow my user to be able to run this
without sudo privileges. See `visudo`


# Running 
## Development Mode
Can start the two processes separately:
1. `uv run ./script/start_daemon.sh`
2. `uv run python main.py`

## Always On Mode
Runs the daemon, the main script, and uses a watchdog to reset the internal hub if all audio samples are 0

```bash
uv run python run_reachy_supervisor.py
```
This is hard coded to the parameters of Frisket currently.


## To Do:
 - Keep track of history and provide between conversations
 - (code) Clean up async handling - it's brittle now, should use a main coroutine "receive events" and have it cancel the others

Other Ideas:
 - Look based on direction of arrival
 - Face tracking + scan (look at you) (run in separate process)
 - Link / embed some of the existing apps (20 questions, hand tracking)