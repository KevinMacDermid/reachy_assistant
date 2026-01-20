# Reachy Mini
This is a small project to explore the Reachy Mini robot.

To Do:
 - Trigger Voice ConversationL
   - See conversation app, openai_realtime ~360. Need to get each delta and put it in a queue for the robot to emit

Other Ideas:
 - Make it possible to cancel emotion moves
 - Look based on direction of arrival
 - Face tracking + scan (look at you) (run in separate process)
 - Link / embed some of the existing apps (20 questions, hand tracking)

# Installation
## Install PortAudio
This is required for OpenWakeWord, and is not installed automatically.

```bash
sudo apt install portaudio19-dev
```

## Install Python-SDK
https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md

That includes instructions for allowing access to the USB

## Install Dependencies
Run 
```
uv sync
```

# Running Daemon
HuggingFace provides a daemon that provides API for the robot.
```bash
uv run reachy-mini-daemon
```

# Wake Word
Working on this - need to find and run some existing voice model
https://huggingface.co/spaces/luisomoreau/hey_reachy_wake_word_detection

# Conversation
Looks like they have conversation with face tracking already
https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/tree/main/src/reachy_mini_conversation_app

https://platform.openai.com/docs/guides/voice-agents?voice-agent-architecture=chained


# Issues
## Audio Troubles
Sometimes have trouble getting audio from the robot, not sure the problem yet,
but saw that if an error is raised by the daemon it can be because the audio device
is already claimed, you can check using
```bash
arecord -l
```
### Issue 1: Failed Conversation App
Tested on Megabyte using conversation app, it also doesn't get or play audio from the robot
- Startup sound does play, sometimes had to adjust volume for it to work
- Unplug and replug USB, now the microphone looks good in settings, and conversation app works.
- **Do not leave the sound window open, Pipewire will hog the device**. Can tell if this happens as there will be
a warning

Once this is fixed the converation app is working fine.

### Issue 2: Failed Audio Sample
When I collect samples using robot.media.get_audio_sample(), I get samples, but they're all 0.0
Evidence:
 - Checked the mic on Pipewire, it's also reading no input.
 - Tried conversation app, it's also no longer getting input

**I think the robot can get in a state where the audio isn't working, not clear if that's
triggered by software, or maybe the USB hub.**

**Can happen from software, unplug it and plug it back in seems to fix it**

Tested raw samples from driver with `arecord -D plughw:0,0 -d 5 -f S16_LE -r 16000 -t raw | od -x`,
it's getting all zeroes.

On Frisket, turned off USB suspend for this device using 
`echo "on" | sudo tee /sys/bus/usb/devices/3-3/power/control`
will see if it's caused by suspend or not.


## Direction of Arrival Issue
Had problem with USB permissions, had to update /etc/udev/rules.d/99-reachy-mini.rules to
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"
# Target the raw USB hardware (usb) - This fixes the USBError Errno 13
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
# Do the same for the Audio/Camera if they are causing issues
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", MODE="0666", GROUP="dialout"
```
The last line was the one that seemed to fix it.