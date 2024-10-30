import tempfile
import contextlib
from pathlib import Path

import pytest

import pandas as pd
from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.scales import (
    get_row_iterator,
    row_processor,
    get_all_scales,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)
from util import no_output


def test_generate_scales_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "scales"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                scales=get_all_scales()[:2],
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

        # 2 scales, 2 play styles, 5 instruments
        assert dataset_df.shape == (2 * 2 * 5, 12)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "root_note_name",
            "mode",
            "play_style",
            "play_style_name",
            "midi_program_num",
            "midi_program_name",
            "midi_category",
            "midi_file_path",
            "synth_file_path",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embeddings_and_train_for_small_scales_dataset() -> None:
    conda_env_name = "syntheory"
    model_configs = [
        {
            "model_name": "MELSPEC",
            "model_type": "MELSPEC",
            "minimum_duration_in_sec": 4,
        },
    ]
    dataset_name = "scales"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "scales"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                scales=get_all_scales(),
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
            == "0529a4eb229be11b59d8507c7808c9a5a4f42542f8891abd97e75e9bde83db24"
        )
        assert dataset_df.shape[0] == (7 * 12 * 2 * 2)

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
        assert metrics == {
            "accuracy": 0.07999999821186066,
            "f1": 0.07999999821186066,
            "loss": 1.8940163850784302,
            "primary": 0.07999999821186066,
            "primary_eval_metric": 0.07999999821186066,
        }
