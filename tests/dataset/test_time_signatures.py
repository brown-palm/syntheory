import tempfile
import contextlib
from pathlib import Path

import pytest
import pandas as pd

from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.time_signatures import (
    CLICK_CONFIGS,
    get_all_time_signatures,
    get_row_iterator,
    row_processor,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)
from util import no_output


def test_generate_time_signatures_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write small subset of real configuration
        dataset_name = "time_signatures"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=Path(tmp_dir),
            row_iterator=get_row_iterator(
                list(get_all_time_signatures())[:2],
                click_configs=CLICK_CONFIGS[:2],
                num_reverb_levels=1,
                num_random_offsets=2,
                target_duration_per_sample_in_sec=4.0,
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            dataset_df = dataset_writer.create_dataset()

        # check the resulting info csv / dataframe
        num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]
        assert num_silent_samples == 0

        # 2 time signatures, 2 click configs, 2 random offsets, 1 reverb level
        assert dataset_df.shape == (2 * 2 * 2, 15)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "time_signature",
            "time_signature_beats",
            "time_signature_subdivision",
            "is_compound",
            "bpm",
            "click_config_name",
            "midi_program_num",
            "midi_file_path",
            "synth_file_path",
            "offset_file_path",
            "offset_time",
            "synth_soundfont",
            "reverb_level",
            "is_silent",
        ]


def test_create_embedding_for_small_time_signature_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "MUSICGEN_DECODER_LM_S",
            "model_type": "MUSICGEN_DECODER_LM_S",
            "minimum_duration_in_sec": 4,
        },
        {
            "model_name": "MELSPEC",
            "model_type": "MELSPEC",
            "minimum_duration_in_sec": 4,
        },
    ]
    all_time_signatures = list(get_all_time_signatures())[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "time_signatures"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                all_time_signatures,
                click_configs=CLICK_CONFIGS[:2],
                num_reverb_levels=1,
                num_random_offsets=2,
                target_duration_per_sample_in_sec=4.0,
                seed=100,
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the contents of the dataset csv file
        assert (
            compute_checksum(dataset_writer.info_csv_filepath)
            == "fe386234facc0381f20f7455ed9c66371ba049d1589de895a8c325b36fbcc8e4"
        )

        # 8 total samples for which we must extract embeddings
        assert dataset_df.shape[0] == 8
        max_samples_per_shard = 2

        # --- DATASET EXTRACTION ---
        for model_config in model_configs:
            # extracts embeddings to files within the temporary test directory
            extract_embeddings_for_model_config_mock_subprocess(
                model_config, dataset_writer, max_samples_per_shard
            )

        # --- CHECK EMBEDDINGS PROPERTIES ---
        # MUSICGEN_DECODER_LM_S
        model_config = model_configs[0]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # 8 samples, 25 layers, of dimension 1024
        assert embeddings_array.shape == (dataset_df.shape[0], 25, 1024)
        assert dataset_coordinator.model_name == "MUSICGEN_DECODER_LM_S"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # MELSPEC
        model_config = model_configs[1]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # 8 samples, no layers, each melspectrogram is of dimension 768
        assert embeddings_array.shape == (dataset_df.shape[0], 768)
        assert dataset_coordinator.model_name == "MELSPEC"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # --- USE EMBEDDINGS IN TRAINING A PROBE ---
        # train a probe on the musicgen embeddings
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MUSICGEN_DECODER_LM",
                "model_size": "S",
                "model_layer": 10,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
                "dropout_p": 0.0,
                "l2_weight_decay": 0.00001,
                "learning_rate": 0.001,
                "num_classes": len(all_time_signatures),
            },
            dataset_writer,
            show_logs=False,
        )
        assert metrics == {
            "accuracy": 0.0,
            "confusion_matrix": [[0, 0], [1, 0]],
            "f1": 0.0,
            "loss": 1.1415386199951172,
            "primary": 0.0,
            "primary_eval_metric": 0.0,
        }

        # train a probe on the melspec
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MELSPEC",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
                "dropout_p": 0.0,
                "l2_weight_decay": 0.00001,
                "learning_rate": 0.001,
                "num_classes": len(all_time_signatures),
            },
            dataset_writer,
            show_logs=False,
        )
        assert metrics == {
            "accuracy": 0.0,
            "confusion_matrix": [[0, 0], [1, 0]],
            "f1": 0.0,
            "loss": 0.8273748159408569,
            "primary": 0.0,
            "primary_eval_metric": 0.0,
        }
