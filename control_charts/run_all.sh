#!/bin/bash
set -e
cd /home/jproanio/Desktop/Modelo_VAE/control_charts
PY=/home/jproanio/venv311/bin/python
for a in "Trojan" "Torii" "Hakai" "Hide and seek" "Muhstik" "Gafyt" "Okiru" "Mirai" "Linux Hajime" "Linux Mirai" "IRCbot" "Kenjiro"; do
  echo "=== $(date '+%H:%M:%S') starting $a ==="
  $PY phase2.py --attack "$a"
  echo "=== $(date '+%H:%M:%S') finished $a ==="
done
echo "=== ALL DONE ==="
