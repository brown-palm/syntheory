import tempfile
import contextlib
from pathlib import Path

import pytest

import pandas as pd
import torch

from dataset.synthetic.dataset_writer import DatasetWriter
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.chord_progressions import (
    PROGRESSIONS,
    get_row_iterator,
    row_processor,
    get_all_keys,
)
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import DatasetEmbeddingInformation
from probe.probes import ProbeExperiment
from tests.probe.util import (
    extract_embeddings_for_model_config_mock_subprocess,
    run_probe_and_return_metrics,
)
from util import no_output


def test_generate_chord_progressions_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # write small subset of real configuration
        dataset_name = "chord_progressions"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                progressions=PROGRESSIONS,
                keys=get_all_keys()[:2],
                instruments=get_instruments(
                    ignore_atonal=True,
                    ignore_polyphonic=True,
                    ignore_highly_articulate=True,
                    take_only_first_category=False,
                )[:1],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the resulting info csv / dataframe
        num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]
        assert num_silent_samples == 0

        # k chord progressions, 2 keys, 1 instrument
        assert dataset_df.shape == (len(PROGRESSIONS) * 2 * 1, 11)
        assert dataset_df.columns.to_list() == [
            "row_id",
            "key_note_name",
            "key_note_pitch_class",
            "chord_progression",
            "midi_program_num",
            "midi_program_name",
            "midi_category",
            "midi_file_path",
            "synth_file_path",
            "synth_soundfont",
            "is_silent",
        ]


def test_create_embeddings_and_train_for_small_chord_progressions_dataset() -> None:
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
        dataset_name = "chord_progressions"
        dataset_writer = DatasetWriter(
            dataset_name=dataset_name,
            save_to_parent_directory=tmp_path,
            row_iterator=get_row_iterator(
                progressions=PROGRESSIONS,
                keys=get_all_keys(),
                instruments=get_instruments(
                    ignore_atonal=True,
                    ignore_polyphonic=True,
                    ignore_highly_articulate=True,
                    take_only_first_category=False,
                )[:1],
            ),
            row_processor=row_processor,
            max_processes=8,
        )
        with no_output():
            dataset_df = dataset_writer.create_dataset()

        # check the contents of the dataset csv file
        assert (
            compute_checksum(dataset_writer.info_csv_filepath)
            == "7b1e179d424977640bb65a806b5ff9486980be9c6b7848edc6b3899bc4b0a5a2"
        )
        assert dataset_df.shape[0] == (len(PROGRESSIONS) * 12 * 1)

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
        metrics, probe_exp = run_probe_and_return_metrics(
            {
                "model_type": "MELSPEC",
                "model_size": "L",
                "model_layer": 0,
                "concept": dataset_name,
                "batch_size": 32,
                "data_standardization": False,
            },
            dataset_writer,
            with_confusion_matrix=False,
            show_logs=False,
            return_probe_experiment_object=True,
        )
        assert metrics == {
            "accuracy": 0.0,
            "f1": 0.0,
            "loss": 3.543002128601074,
            "primary": 0.0,
            "primary_eval_metric": 0.0,
        }

        assert isinstance(probe_exp, ProbeExperiment)

        # test saving the probe to disk and loading it
        uid, root_dir = probe_exp.save(tmp_path)

        # this is the model UID, not the dataset hash
        assert uid == "cec4297b8c3439d8ff570717a4be956a6924b6fa"

        # load same model from disk
        loaded_probe = ProbeExperiment.load(uid, root_dir)
        assert isinstance(loaded_probe, ProbeExperiment)
        for a, b in zip(
            probe_exp.probe.state_dict().items(),
            loaded_probe.probe.state_dict().items(),
        ):
            assert torch.equal(b[1], a[1])

        assert probe_exp.cfg == loaded_probe.cfg

        # clean up
        loaded_probe.delete()
