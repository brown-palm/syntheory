import tempfile
import contextlib
from pathlib import Path

import pandas as pd
import pytest

from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.tempos import (
    CLICK_CONFIGS,
    get_all_tempos,
    get_row_iterator,
    row_processor,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from util import no_output

from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)


def test_get_all_tempos() -> None:
    tempos = list(get_all_tempos(50, 100))
    assert tempos[0] == 50
    assert tempos[-1] == 100


def test_generate_tempos_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write small subset of real configuration
        dataset_writer = DatasetWriter(
            dataset_name="tempos",
            save_to_parent_directory=Path(tmp_dir),
            row_iterator=get_row_iterator(
                slowest_bpm=70,
                fastest_bpm=79,
                click_configs=CLICK_CONFIGS[:2],
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

        # 10 BPMs, 2 click configs, 2 random offsets
        assert dataset_df.shape == (10 * 2 * 2, 10)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "bpm",
            "click_config_name",
            "midi_program_num",
            "midi_file_path",
            "synth_file_path",
            "offset_file_path",
            "offset_time",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embedding_for_small_tempos_signature_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "JUKEBOX",
            "model_type": "JUKEBOX",
            "minimum_duration_in_sec": 4,
        },
        {
            "model_name": "MFCC",
            "model_type": "MFCC",
            "minimum_duration_in_sec": 4,
        },
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_name = "tempos"

        # write small subset of real configuration
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=Path(tmp_dir),
            row_iterator=get_row_iterator(
                slowest_bpm=80,
                fastest_bpm=99,
                click_configs=CLICK_CONFIGS[:2],
                num_random_offsets=2,
                target_duration_per_sample_in_sec=4.0,
                seed=100,
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            dataset_df = dataset_writer.create_dataset()

        # check the contents of the dataset csv file
        assert (
            compute_checksum(dataset_writer.info_csv_filepath)
            == "eddf9f9fca1ac59dcb0f735f67e96bc62e501961a04201707fb82bb5a08efe8c"
        )

        # 20 BPMs * 2 clicks * 2 offsets (80)
        assert dataset_df.shape[0] == (20 * 2 * 2)

        # each shard should do 32 samples
        max_samples_per_shard = 32

        # --- DATASET EXTRACTION ---
        for model_config in model_configs:
            # extracts embeddings to files within the temporary test directory
            extract_embeddings_for_model_config_mock_subprocess(
                model_config, dataset_writer, max_samples_per_shard
            )

        # --- CHECK EMBEDDINGS PROPERTIES ---
        # JUKEBOX
        model_config = model_configs[0]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # 80 samples, 72 layers, of dimension 4800
        assert embeddings_array.shape == (dataset_df.shape[0], 72, 4800)
        assert dataset_coordinator.model_name == "JUKEBOX"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # MFCC
        model_config = model_configs[1]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()
        # 80 samples, no layers, each MFCC is of dimension 120
        assert embeddings_array.shape == (dataset_df.shape[0], 120)
        assert dataset_coordinator.model_name == "MFCC"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # --- USE EMBEDDINGS IN TRAINING A PROBE ---
        # train a probe on the jukebox embeddings
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "JUKEBOX",
                "model_size": "L",
                "model_layer": 10,
                "concept": dataset_name,
            },
            dataset_writer,
            show_logs=False,
        )
        assert metrics == {
            "loss": 1.547699213027954,
            "primary": -9.461006164550781,
            "primary_eval_metric": -9.461006164550781,
            "r2": -9.461006164550781,
        }

        # train a probe on the mfcc
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MFCC",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
            },
            dataset_writer,
            show_logs=False,
        )
        assert metrics == {
            "loss": 0.09626750648021698,
            "primary": 0.34932124614715576,
            "primary_eval_metric": 0.34932124614715576,
            "r2": 0.34932124614715576,
        }
