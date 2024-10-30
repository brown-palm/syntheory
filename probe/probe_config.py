import json
from embeddings.config_checksum import compute_checksum


CONCEPT_LABELS = {
    "chord_progressions": [
        (19, "chord_progression"),
        (12, "key_note_name"),
    ],
    "chords": [(4, "chord_type"), (3, "inversion"), (12, "root_note_name")],
    "scales": [(7, "mode"), (12, "root_note_name")],
    "intervals": [(12, "interval"), (12, "root_note_name")],
    "notes": [(12, "root_note_pitch_class"), (9, "octave")],
    "time_signatures": [
        (8, "time_signature"),
        (6, "time_signature_beats"),
        (3, "time_signature_subdivision"),
    ],
    "tempos": [(161, "bpm")],
}


# hyperparameter grid used in Jukemir paper
HYPERPARAMS = {
    "data_standardization": [False, True],
    "hidden_layer_sizes": [[], [512]],
    "batch_size": [64, 256],
    "learning_rate": [1e-5, 1e-4, 1e-3],
    "dropout_p": [0.25, 0.5, 0.75],
    "l2_weight_decay": [None, 1e-4, 1e-3],
}


SWEEP_CONFIGS = {
    "handcrafted": {
        "wandb_sweep_parameters": {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "primary_eval_metric"},
            "parameters": {
                "model_type": {"values": ["CHROMA", "MELSPEC", "MFCC", "HANDCRAFT"]},
                "model_size": {"values": ["L"]},
                "model_layer": {"values": [0]},
                "concept": {"values": list(CONCEPT_LABELS.keys())},
                **{
                    x: {"values": y} for x, y in HYPERPARAMS.items()
                },  # hyperparameter search
            },
        },
        "wandb_project_name": "music-theory-handcrafted",
    },
    "encoder": {
        "wandb_sweep_parameters": {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "primary_eval_metric"},
            "parameters": {
                "model_type": {"values": ["MUSICGEN_AUDIO_ENCODER"]},
                "model_size": {"values": ["L"]},
                "model_layer": {"values": [0]},
                "concept": {"values": list(CONCEPT_LABELS.keys())},
                **{
                    x: {"values": y} for x, y in HYPERPARAMS.items()
                },  # hyperparameter search
            },
        },
        "wandb_project_name": "music-theory-encoder",
    },
    "jukebox": {
        "wandb_sweep_parameters": {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "primary_eval_metric"},
            "parameters": {
                "model_type": {"values": ["JUKEBOX"]},
                "model_size": {"values": ["L"]},
                "model_layer": {"values": list(range(72))},
                "concept": {"values": list(CONCEPT_LABELS.keys())},
            },
        },
        "wandb_project_name": "music-theory-jukebox",
    },
    "musicgen": {
        "wandb_sweep_parameters": {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "primary_eval_metric"},
            "parameters": {
                "model_type": {"values": ["MUSICGEN_DECODER"]},
                "model_size": {"values": ["L", "M", "S"]},
                "model_layer": {"values": list(range(49))},
                "concept": {"values": list(CONCEPT_LABELS.keys())},
            },
        },
        "wandb_project_name": "music-theory-musicgen",
    },
}


class ProbeExperimentConfig(dict):
    """Defines the parameters of the probe experiment to use."""

    _DEFAULTS = {
        "model_hash": None,
        "dataset": None,
        "dataset_embeddings_label_column_name": None,
        "data_standardization": True,
        "hidden_layer_sizes": [],
        "batch_size": 64,
        "learning_rate": 1e-3,
        "dropout_p": 0.5,
        "l2_weight_decay": None,
        "max_num_epochs": None,
        "early_stopping_metric": "primary",
        "early_stopping": True,
        "early_stopping_eval_frequency": 8,
        "early_stopping_boredom": 256,
        "seed": 0,
        "num_outputs": None,
        # if this is true, all the embedding files used in test/train are loaded into RAM
        # otherwise, we load only their location in a zarr file on disk and load as needed
        "load_embeddings_in_memory": False,
    }
    _REQUIRED = [
        "dataset",
        "dataset_embeddings_label_column_name",
        "model_hash",
        # output dimensionality of the probe MLP
        "num_outputs",
    ]

    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(*args, **kwargs)

        for field in self._REQUIRED:
            if field not in kwargs:
                raise ValueError(f"Required field {field} missing")

        for field in kwargs.keys():
            if field not in self._DEFAULTS:
                raise ValueError(f"Unknown field {field} specified")

        # set the defaults
        for field, value in self._DEFAULTS.items():
            if field in kwargs:
                value = kwargs[field]
            self[field] = value

        try:
            json.dumps(self)
        except Exception as e:
            raise ValueError(f"All values must be JSON-serializable. Got error: {e}")

    def uid(self) -> str:
        # hash the config
        return compute_checksum(
            json.dumps(self, indent=2, sort_keys=True).encode("utf-8"), algorithm="sha1"
        )
