#!/bin/bash
set -e

# ensure the path while running this
cd "$(dirname "$0")"

# go to the synth code folder and compile a release build
cd ../midi2audio
cargo build --release