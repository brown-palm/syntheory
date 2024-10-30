import tempfile
import contextlib
from pathlib import Path

import pytest

import pandas as pd
from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.chords import (
    get_row_iterator,
    row_processor,
    get_all_chords,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)
from util import no_output


def test_generate_chords_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "chords"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                chords=get_all_chords()[:10],
                instruments=get_instruments(
                    ignore_atonal=True,
                    ignore_polyphonic=True,
                    ignore_highly_articulate=True,
                    take_only_first_category=False,
                )[:5],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the resulting info csv / dataframe
        num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]
        assert num_silent_samples == 0

        # 10 chord / pitch class combos, 3 inversions for triads, 5 instruments
        assert dataset_df.shape == (10 * 3 * 5, 13)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "root_note_name",
            "chord_type",
            "inversion",
            "root_note_is_accidental",
            "root_note_pitch_class",
            "midi_program_num",
            "midi_program_name",
            "midi_category",
            "midi_file_path",
            "synth_file_path",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embeddings_and_train_for_small_chords_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "MELSPEC",
            "model_type": "MELSPEC",
            "minimum_duration_in_sec": 4,
        },
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "chords"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                chords=get_all_chords(),
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
            == "fd2ef18e2c59606acbf9f404db594e98c5ee2a8a3ddcd3c071947ff82001086c"
        )
        assert dataset_df.shape[0] == (4 * 12 * 3 * 2)

        max_samples_per_shard = 32

        # --- DATASET EXTRACTION ---
        for model_config in model_configs:
            # extracts embeddings to files within the temporary test directory
            extract_embeddings_for_model_config_mock_subprocess(
                model_config, dataset_writer, max_samples_per_shard
            )

        # --- CHECK EMBEDDINGS PROPERTIES ---
        model_config = model_configs[0]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # n samples of dimension 768
        assert embeddings_array.shape == (dataset_df.shape[0], 768)
        assert dataset_coordinator.model_name == "MELSPEC"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # --- USE EMBEDDINGS IN TRAINING A PROBE ---
        # train a probe on the musicgen embeddings
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MELSPEC",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
                "batch_size": 12,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
            show_logs=False,
        )
        # very interesting result
        assert metrics == {
            "accuracy": 0.8372092843055725,
            "f1": 0.8372092843055725,
            "loss": 0.48251622915267944,
            "primary": 0.8372092843055725,
            "primary_eval_metric": 0.8372092843055725,
        }
