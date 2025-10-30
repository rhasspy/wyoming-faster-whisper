#!/usr/bin/env bash
cd /usr/src
.venv/bin/python3 -m wyoming_faster_whisper \
    --uri 'tcp://0.0.0.0:10300' "$@"
