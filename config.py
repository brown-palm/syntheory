"""
Global Project Settings
"""
from pathlib import Path
import yaml
from typing import Dict, Any

# the enclosing folder of the repository
REPO_ROOT = Path(__file__).parent

# put all files, midi, wav, etc. here
OUTPUT_DIR = REPO_ROOT / "data"

# expect synth binary to be here
DEFAULT_SYNTH_BINARY_LOCATION = REPO_ROOT / Path("midi2audio/target/release/midi2audio")

# expect synth soundfont to be here
DEFAULT_SOUNDFONT_LOCATION = REPO_ROOT / Path("midi2audio/data/TimGM6mb.sf2")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)