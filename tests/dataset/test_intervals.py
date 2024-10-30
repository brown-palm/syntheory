import tempfile
import contextlib
from pathlib import Path

import pytest

import pandas as pd
from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.intervals import (
    get_row_iterator,
    row_processor,
    get_all_interval_midi_settings,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)
from util import no_output


def test_generate_intervals_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "intervals"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                intervals=get_all_interval_midi_settings()[:10],
                instruments=get_instruments(
                    ignore_atonal=True,
                    ignore_polyphonic=True,
                    ignore_highly_articulate=True,
                    take_only_first_category=False,
                )[:10],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the resulting info csv / dataframe
        num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]
        assert num_silent_samples == 0

        # 10 interval settings, 10 instruments, 3 play styles
        assert dataset_df.shape == (10 * 10 * 3, 14)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "root_note_name",
            "root_note_pitch_class",
            "interval",
            "play_style",
            "play_style_name",
            "midi_note_val",
            "midi_program_num",
            "midi_program_name",
            "midi_category",
            "midi_file_path",
            "synth_file_path",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embeddings_and_train_for_small_intervals_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "MUSICGEN_DECODER_LM_L",
            "model_type": "MUSICGEN_DECODER_LM_L",
            "minimum_duration_in_sec": 4,
        },
    ]
    dataset_name = "intervals"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "intervals"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                intervals=get_all_interval_midi_settings()[:20],
                instruments=get_instruments(
                    ignore_atonal=True,
                    ignore_polyphonic=True,
                    ignore_highly_articulate=True,
                    take_only_first_category=False,
                )[:2],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the contents of the dataset csv file
        assert (
            compute_checksum(dataset_writer.info_csv_filepath)
            == "9a7203999f4ad937c4ee33eec9d35c53842b46cfb54ed5367cbe2f82886cd4e3"
        )
        assert dataset_df.shape[0] == (20 * 2 * 3)

        max_samples_per_shard = 32

        # --- DATASET EXTRACTION ---
        for model_config in model_configs:
            # extracts embeddings to files within the temporary test directory
            extract_embeddings_for_model_config_mock_subprocess(
                model_config, dataset_writer, max_samples_per_shard
            )

        # --- CHECK EMBEDDINGS PROPERTIES ---
        # MUSICGEN_DECODER_LM_L
        model_config = model_configs[0]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # 80 samples, 49 layers, of dimension 2048
        assert embeddings_array.shape == (dataset_df.shape[0], 49, 2048)
        assert dataset_coordinator.model_name == "MUSICGEN_DECODER_LM_L"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # --- USE EMBEDDINGS IN TRAINING A PROBE ---
        # train a probe on the musicgen embeddings
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MUSICGEN_DECODER_LM",
                "model_size": "L",
                "model_layer": 10,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
        )
        assert metrics == {
            "accuracy": 0.1666666716337204,
            "f1": 0.1666666716337204,
            "loss": 25.022735595703125,
            "primary": 0.1666666716337204,
            "primary_eval_metric": 0.1666666716337204,
        }
