#!/bin/bash
set -e
cd /home/jproanio/Desktop/Modelo_VAE/control_charts
PY=/home/jproanio/venv311/bin/python
for a in "Hakai" "Hide and seek" "Muhstik" "Okiru" "Mirai" "Linux Hajime" "Linux Mirai" "IRCbot" "Kenjiro"; do
  echo "=== $(date '+%H:%M:%S') starting $a ==="
  $PY phase2.py --attack "$a"
  echo "=== $(date '+%H:%M:%S') finished $a ==="
done
echo "=== ALL DONE ==="
