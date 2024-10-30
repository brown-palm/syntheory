from pathlib import Path
from config import load_config


def test_load_config() -> None:
    # test loading a yaml file
    yaml_to_laod = Path(__file__).parent / "config_example.yaml"
    loaded_config = load_config(str(yaml_to_laod.absolute()))
    assert loaded_config == {
        "concepts": [
            "time_signatures",
            "tempos",
            "notes",
            "chords",
            "scales",
            "intervals",
            "chord_progressions",
        ],
        "models": [
            "HANDCRAFT",
            "MUSICGEN_AUDIO_ENCODER",
            "JUKEBOX",
            "MUSICGEN_DECODER_LM_S",
            "MUSICGEN_DECODER_LM_M",
            "MUSICGEN_DECODER_LM_L",
            "MELSPEC",
            "MFCC",
            "CHROMA",
        ],
        "settings": {
            "conda_env_name": "syntheory",
            "max_samples_per_shard": 300,
            "minimum_duration_in_sec": 4,
            "slurm_partition": "gpu",
        },
    }
