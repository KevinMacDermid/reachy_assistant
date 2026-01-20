#!/bin/bash
echo "If this all zeroes then the audio driver is not working"
arecord -D plughw:0,0 -d 5 -f S16_LE -r 16000 -t raw | od -x