import tempfile
import contextlib
from pathlib import Path

import pytest

import pandas as pd
from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.notes import (
    get_row_iterator,
    row_processor,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)


def test_generate_notes_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        num_notes = 12
        dataset_writer = DatasetWriter(
            dataset_name="notes",
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                range(50, 50 + num_notes),
                list(
                    get_instruments(
                        ignore_atonal=True,
                        ignore_polyphonic=True,
                        ignore_highly_articulate=True,
                        take_only_first_category=False,
                    )
                )[:1],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            dataset_df = dataset_writer.create_dataset()

        # check the resulting info csv / dataframe
        num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]
        assert num_silent_samples == 0

        # 12 notes
        assert dataset_df.shape == (num_notes, 14)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "root_note_name",
            "root_note_pitch_class",
            "octave",
            "root_note_is_accidental",
            "register",
            "midi_note_val",
            "midi_program_num",
            "midi_program_name",
            "midi_category",
            "midi_file_path",
            "synth_file_path",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embeddings_and_train_for_small_notes_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "MUSICGEN_DECODER_LM_M",
            "model_type": "MUSICGEN_DECODER_LM_M",
            "minimum_duration_in_sec": 4,
        },
        {
            "model_name": "CHROMA",
            "model_type": "CHROMA",
            "minimum_duration_in_sec": 4,
        },
        {
            "model_name": "HANDCRAFT",
            "model_type": "HANDCRAFT",
            "minimum_duration_in_sec": 4,
        },
    ]
    dataset_name = "notes"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        num_notes = 24
        num_instruments = 2
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                range(50, 50 + num_notes),
                list(
                    get_instruments(
                        ignore_atonal=True,
                        ignore_polyphonic=True,
                        ignore_highly_articulate=True,
                        take_only_first_category=False,
                    )
                )[:num_instruments],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            dataset_df = dataset_writer.create_dataset()

        # check the contents of the dataset csv file
        assert (
            compute_checksum(dataset_writer.info_csv_filepath)
            == "2d4d02f581cfed960a3b2a30b9343bb04c42eb1c42fffd83ca3198a1d0e22e29"
        )
        assert dataset_df.shape[0] == (num_notes * num_instruments)

        # check shard sizes that don't evenly divide the total
        max_samples_per_shard = 7

        # --- DATASET EXTRACTION ---
        for model_config in model_configs:
            # extracts embeddings to files within the temporary test directory
            extract_embeddings_for_model_config_mock_subprocess(
                model_config, dataset_writer, max_samples_per_shard
            )

        # --- CHECK EMBEDDINGS PROPERTIES ---
        # MUSICGEN_DECODER_LM_M
        model_config = model_configs[0]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()

        # 80 samples, 49 layers, of dimension 1536
        assert embeddings_array.shape == (dataset_df.shape[0], 49, 1536)
        assert dataset_coordinator.model_name == "MUSICGEN_DECODER_LM_M"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # CHROMA
        model_config = model_configs[1]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()
        # 12 samples, no layers, each CHROMA is of dimension 72
        assert embeddings_array.shape == (dataset_df.shape[0], 72)
        assert dataset_coordinator.model_name == "CHROMA"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # HANDCRAFT
        model_config = model_configs[2]
        dataset_coordinator = DatasetEmbeddingInformation(
            dataset_writer.dataset_path, model_config, max_samples_per_shard
        )
        embeddings_array = dataset_coordinator.get_or_create_zarr_file()
        # 12 samples, no layers, each HANDCRAFT is the concatenation of MFCC, MELSPEC, and CHROMA and is
        # of dimension 960
        assert embeddings_array.shape == (dataset_df.shape[0], 960)
        assert dataset_coordinator.model_name == "HANDCRAFT"
        with pytest.raises(RuntimeError) as exc:
            dataset_coordinator.make_status_folder()
            assert str(exc.value).startswith("Status folder already exists")

        # --- USE EMBEDDINGS IN TRAINING A PROBE ---
        # train a probe on the musicgen embeddings
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "MUSICGEN_DECODER_LM",
                "model_size": "M",
                "model_layer": 10,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
        )
        assert metrics == {
            "accuracy": 0.0,
            "f1": 0.0,
            "loss": 2.631864547729492,
            "primary": 0.0,
            "primary_eval_metric": 0.0,
        }
        # train a probe on the chroma features
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "CHROMA",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
        )
        assert metrics == {
            "accuracy": 1.0,
            "f1": 1.0,
            "loss": 0.1975461095571518,
            "primary": 1.0,
            "primary_eval_metric": 1.0,
        }

        # train a probe on the concat handcrafted features
        metrics = run_probe_and_return_metrics(
            {
                "model_type": "HANDCRAFT",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
                "batch_size": 2,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
        )
        assert metrics == {
            "accuracy": 0.5714285969734192,
            "f1": 0.5714285969734192,
            "loss": 24.545541763305664,
            "primary": 0.5714285969734192,
            "primary_eval_metric": 0.5714285969734192,
        }
