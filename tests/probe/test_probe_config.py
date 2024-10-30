import pytest
from decimal import Decimal
from probe.probe_config import ProbeExperimentConfig


def test_probe_experiment_config_class() -> None:
    cfg = ProbeExperimentConfig(
        dataset_embeddings_label_column_name="something",
        dataset="test",
        num_outputs=10,
        model_hash="xyz",
        max_num_epochs=100,
        seed=42,
        load_embeddings_in_memory=True,
    )

    # the default value should exist without explicit setting
    assert cfg["batch_size"] == 64

    # test hashing of settings
    assert cfg.uid() == "23f4db03a995b56bc70d62dd5453670c68ea8654"

    with pytest.raises(ValueError) as exc:
        ProbeExperimentConfig(
            dataset_embeddings_label_column_name="something",
            num_outputs=10,
            model_hash="xyz",
            max_num_epochs=100,
            seed=42,
            load_embeddings_in_memory=True,
        )
        assert str(exc.value) == "Required field dataset missing"

    with pytest.raises(ValueError) as exc:
        ProbeExperimentConfig(
            dataset_embeddings_label_column_name="something",
            dataset="test",
            num_outputs=10,
            model_hash="xyz",
            max_num_epochs=100,
            seed=42,
            load_embeddings_in_memory=True,
            bad_param="something",
        )
        assert str(exc.value) == "Unknown field bad_param specified"

    with pytest.raises(ValueError) as exc:
        ProbeExperimentConfig(
            dataset_embeddings_label_column_name="something",
            dataset="test",
            num_outputs=10,
            model_hash="xyz",
            max_num_epochs=100,
            # Decimal object is not json serializable
            seed=Decimal(42.0),
            load_embeddings_in_memory=True,
        )
        assert str(exc.value).startswith("All values must be JSON-serializable.")
