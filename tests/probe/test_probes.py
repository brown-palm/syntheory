from typing import Any, Dict
from pathlib import Path
from collections import namedtuple
import contextlib
from unittest.mock import patch, Mock, MagicMock
import pytest
import zarr
import pandas as pd
import numpy as np

from probe.main import start
from tests.probe.util import MockWandbConfig, run_probe_and_return_metrics, ModelConfigurationDoesNotExist


def test_invalid_configuration() -> None:
    # this configuration doesn't exist, so the helper function will fail to fetch its
    # metrics (because it failed to start)
    with pytest.raises(ModelConfigurationDoesNotExist):
        run_probe_and_return_metrics(
            {
                "model_type": "JUKEBOX",
                # there is no M Jukebox model
                "model_size": "M",
                "model_layer": 10,
                "concept": "tempos",
            }
        )


def test_regression_probe() -> None:
    try:
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "JUKEBOX",
                "model_size": "L",
                "model_layer": 38,
                "concept": "tempos",
            },
            show_logs=False,
        )
        assert metrics["r2"] == 0.9916887879371643
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "JUKEBOX",
                "model_size": "L",
                "model_layer": 60,
                "concept": "tempos",
                "label_column_name": "bpm",
                "dropout_p": 0.5,
                "learning_rate": 0.00001,
                "l2_weight_decay": 0.001,
                "data_standardization": False,
            },
            show_logs=False,
        )
        assert metrics["r2"] == -0.3672337532043457
    except ModelConfigurationDoesNotExist:
        pass


def test_musicgen_audio_encoder_probe() -> None:
    try: 
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MUSICGEN_AUDIO_ENCODER",
                "model_size": "L",
                "model_layer": 0,
                "concept": "notes",
                "batch_size": 256,
                "data_standardization": False,
                "dropout_p": 0.0,
                "l2_weight_decay": 0.00001,
                "learning_rate": 0.001,
            },
            show_logs=False,
        )
        confusion_matrix = metrics["confusion_matrix"]
        assert isinstance(confusion_matrix, list)
        assert len(confusion_matrix) == 12
        assert len(confusion_matrix[0]) == 12
    except ModelConfigurationDoesNotExist:
        pass
