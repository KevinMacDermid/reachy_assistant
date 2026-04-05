#!/bin/bash
# Installs Gstreamer on old Ubuntu
sudo add-apt-repository ppa:savoury1/multimedia
sudo apt update
sudo apt-get install -y \
  libgstreamer-plugins-bad1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer1.0-dev \
  libglib2.0-dev \
  libssl-dev \
  libgirepository1.0-dev \
  libcairo2-dev \
  libportaudio2 \
  libnice10 \
  gstreamer1.0-plugins-good \
  gstreamer1.0-alsa \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-nice \
  python3-gi \
  python3-gi-cairo