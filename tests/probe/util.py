from typing import Any, Dict, List, Optional, Union, Tuple
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import math

import numpy as np

from dataset.synthetic.dataset_writer import DatasetWriter
from embeddings.config_checksum import compute_checksum
from embeddings.extract_embeddings import (
    get_scripts_to_extract_embeddings_for_dataset_with_model,
)
from embeddings.embeddings_cli import extract_shard
from probe.main import start
from probe.probes import ProbeExperiment
from util import no_output

class ModelConfigurationDoesNotExist(Exception):
    pass


class MockWandbConfig:
    def __init__(self, sweep_attrs: Dict[str, Any]) -> None:
        for k, v in sweep_attrs.items():
            setattr(self, k, v)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"No attribute named: {name}")

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise KeyError(f"No attribute named: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value


def extract_embeddings_for_model_config_mock_subprocess(
    model_config, dataset_writer: DatasetWriter, max_samples_per_shard: int = 32
) -> List[Path]:
    model_config_checksum = compute_checksum(model_config)
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.stdout = "Success".encode("utf-8")
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # run dataset extractor code, this would typically create
        # shell scripts that can be enqueued to a compute cluster
        shard_scripts = get_scripts_to_extract_embeddings_for_dataset_with_model(
            dataset_writer.dataset_path,
            model_config,
            conda_env_name="syntheory",
            slurm_partition="gpu",
            max_samples_per_shard=max_samples_per_shard,
        )
        assert len(shard_scripts) == math.ceil(
            dataset_writer.get_dataset_as_pandas_dataframe().shape[0]
            / max_samples_per_shard
        )

        # test that after each shard is done, its assigned indexes not all zero in the zarr array
        for shard_idx in range(len(shard_scripts)):
            # extract embeddings to indices assigned to this shard
            zarr_file, idxs_written = extract_shard(
                dataset_writer.dataset_name,
                shard_idx,
                model_config_checksum,
                root_dir=dataset_writer.parent_directory,
            )

            # check that they were written correctly
            axis = tuple(range(1, len(zarr_file.shape)))
            batch = zarr_file[idxs_written]
            num_blank_rows = np.sum(np.all(batch == 0, axis=axis))
            assert num_blank_rows == 0

        return shard_scripts


def run_probe_and_return_metrics(
    probe_train_run_config,
    dataset_writer: Optional[DatasetWriter] = None,
    with_confusion_matrix: bool = True,
    show_logs: bool = True,
    metric_group: str = "valid",
    return_probe_experiment_object: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict, str, Any], ProbeExperiment]:
    ctx = nullcontext if show_logs else no_output
    conf = MockWandbConfig(probe_train_run_config)
    with patch("probe.main.wandb") as mock_wandb:
        mock_wandb.config = conf
        mock_wandb.init = lambda: print("init")
        mock_wandb.log = MagicMock()

        kwargs = {
            "use_wandb": False,
            "random_seed": 100,
        }
        if dataset_writer:
            kwargs["base_path_parent"] = dataset_writer.dataset_path.parent

        with ctx():
            exp_result = start(**kwargs)
            
            if exp_result is None:
                # the configuration did not exist on disk. Some of the older tests
                # assume embedding folders exist already - if we encounter one of 
                # those tests and there is no folder that we need, raise an error.
                # The tests that do this will catch that error. 
                raise ModelConfigurationDoesNotExist(f"No configuration existed for {probe_train_run_config}.")

            metrics = exp_result.eval(
                metric_group, with_confusion_matrix=with_confusion_matrix
            )
            if return_probe_experiment_object:
                # return metrics and the probe experimentobject
                return metrics, exp_result
            else:
                # return only metrics
                return metrics
