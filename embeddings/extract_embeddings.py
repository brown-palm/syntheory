import argparse
import datetime
import time
from typing import NamedTuple, Dict, Any, List, Optional, Tuple
import json
import subprocess
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio
import zarr

from util import use_770_permissions
from config import OUTPUT_DIR, load_config
from embeddings.config_checksum import compute_checksum
from embeddings.models import audio_file_to_embedding_np_array, Model, load_musicgen_model

class ShardStatus(Enum):
    DONE = 1
    FAIL = 2
    IN_PROGRESS = 3

JOB_HOURS_FAILURE_THRESHOLD = 6

SLURM_JOB_BASE = r"""
#!/bin/bash
#SBATCH -p {slurm_partition} --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J extract_embeddings_{hash}_shard_{dataset_shard}_{dataset_folder_name}
#SBATCH -e logs/embeddings/extract_embeddings_{hash}_shard_{dataset_shard}_{dataset_folder_name}.err
#SBATCH -o logs/embeddings/extract_embeddings_{hash}_shard_{dataset_shard}_{dataset_folder_name}.out

# Activate virtual environment
conda activate {conda_env_name}

# Model Configuration:
{model_config_str}

# Run the script
python embeddings/embeddings_cli.py --dataset_folder_name {dataset_folder_name} --dataset_shard {dataset_shard} --model_config_checksum {hash}
"""


def run_shell_script(script_path: str) -> None:
    try:
        result = subprocess.run(
            script_path,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Script output:", result.stdout.decode())
        print("Return code:", result.returncode)
        if result.returncode != 0:
            raise RuntimeError("The shard running script has errors. ")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr.decode()}")
        raise


def get_audio_file_path_from_sample_info(sample_info: Dict[str, Any]) -> str:
    # use the offset sample if it exists
    synth_filepath = sample_info["synth_file_path"]
    offset_filepath = sample_info.get("offset_file_path")
    audio_filepath = offset_filepath or synth_filepath
    return audio_filepath



class DatasetEmbeddingInformation:

    def __init__(self, dataset_folder: Path, model_config: Dict[str, Any], max_samples_per_shard: int):
        self.dataset_folder = dataset_folder
        self.model_config = model_config
        self.model_config_checksum = compute_checksum(model_config)
        self.model_name = model_config["model_name"]
        self.max_samples_per_shard = max_samples_per_shard
        self.dataset_name = self.dataset_folder.parts[-1]
        self.dataset_info_df = pd.read_csv(self.dataset_folder / "info.csv")
        self.num_total_samples = self.dataset_info_df.shape[0]
        self.status_folder = self.dataset_folder / (
            self.dataset_name + f"_{self.model_config_checksum}_status"
        )

        if not self.status_folder.is_dir():
            self.make_status_folder()

        self.zarr_file_path = self.dataset_folder / (
            "{0}_{1}_{2}.zarr".format(
                self.dataset_name,
                self.model_name,
                self.model_config_checksum
            )
        )
        self.zarr_sync_path = self.dataset_folder / (
            "{0}_{1}_{2}_sync.sync".format(
                self.dataset_name,
                self.model_name,
                self.model_config_checksum
            )
        )
        self.embeddings_info_file = self.dataset_folder / (
            "{0}_{1}_embeddings_info.csv".format(
                self.dataset_name,
                self.model_config_checksum,
            )
        )

        self.model_config_json_path = self.dataset_folder / (
            "{0}_{1}.json".format(
                self.dataset_name,
                self.model_config_checksum
            )
        )
        self.model_config_json_path.write_text(json.dumps(model_config))


    def _get_shard_sizes(self) -> List[int]:
        return DatasetEmbeddingInformation.get_shard_sizes(self.num_total_samples, self.max_samples_per_shard)

    @staticmethod
    def get_shard_sizes(num_total_samples: int, max_samples_per_shard: int) -> List[int]:
        num_full_shards = num_total_samples // max_samples_per_shard
        leftovers = num_total_samples % max_samples_per_shard
        shard_sizes = ([max_samples_per_shard] * num_full_shards)
        if leftovers:
            shard_sizes += [leftovers]
        return shard_sizes

    def load_zarr_file(self) -> Optional[zarr]:
        if not self.zarr_file_path.exists():
            # does not exist
            return None

        # if it exists, load the sync and array files
        zarr_file_sync = zarr.ProcessSynchronizer(
            str(self.zarr_sync_path)
        )
        zarr_file = zarr.open(
            str(self.zarr_file_path),
            mode="a",
            synchronizer=zarr_file_sync,
        )
        # check that the shape matches what we seek to write
        if zarr_file.shape[0] != self.num_total_samples:
            raise RuntimeError(
                f"The embeddings zarr file existed at: {str(self.zarr_file_path.absolute())}, but it was of incorrect dimension: {zarr_file.shape}"
            )
        
        # checks passed, return yes it is valid
        return zarr_file

    def get_or_create_zarr_file(self) -> zarr:
        zarr_file = self.load_zarr_file()

        if zarr_file is not None:
            # nothing to do 
            return zarr_file

        # load model
        model_type = Model[self.model_config["model_type"]]
        if model_type in {
            Model.MUSICGEN_AUDIO_ENCODER,
            Model.MUSICGEN_DECODER_LM_S,
            Model.MUSICGEN_DECODER_LM_M,
            Model.MUSICGEN_DECODER_LM_L,
        }:
            processor, model = load_musicgen_model(model_type)
        else:
            # this is either a hand-crafted feature, or JUKEBOX
            processor, model = None, None

        # embeddings file does not yet exist, initialize it to all zeros
        first_sample = self.dataset_info_df.iloc[0].to_dict()
        audio_filepath = get_audio_file_path_from_sample_info(first_sample)

        # do only 1 extraction just to get the dimensions
        embedding = get_embedding_from_model_using_config(
            self.dataset_folder / Path(audio_filepath),
            self.model_config,
            processor,
            model,
        )
        embeddings_shape = embedding.shape

        single_shard_zeros = np.zeros(
            (min(self.max_samples_per_shard, self.dataset_info_df.shape[0]),) + embeddings_shape
        )

        zarr_file_sync = zarr.ProcessSynchronizer(
            str(self.zarr_sync_path.absolute())
        )

        shard_sizes = self._get_shard_sizes()

        # create the first shard
        first_shard_zeros = np.zeros((shard_sizes[0],) + embeddings_shape)

        # need a single shard so we can save it
        zarr.save(str(self.zarr_file_path.absolute()), first_shard_zeros)
        zarr_file = zarr.open(
            str(self.zarr_file_path.absolute()),
            mode="a",
            synchronizer=zarr_file_sync,
        )

        # append each shard to the zarr file. We do this instead of allocating it all 
        # at once because the zero array might not fit in memory
        for shard_size in shard_sizes[1:]:
            if shard_size != first_shard_zeros.shape[0]:
                first_shard_zeros = np.zeros((shard_size,) + embeddings_shape)
            zarr_file.append(first_shard_zeros)

        # check that the zarr file dimension is correct
        assert zarr_file.shape == (self.num_total_samples, *embeddings_shape)

        return zarr_file

    def make_status_folder(self) -> None:
        if self.status_folder.is_dir():
            raise RuntimeError(f"Status folder already exists. Check: {self.make_status_folder}")

        shard_sizes = self._get_shard_sizes()
        self.status_folder.mkdir(parents=True, exist_ok=True)
        p_total_shards = self.status_folder / "total_shards.txt"
        p_total_shards.write_text(str(len(shard_sizes)))

    def get_shard_statuses(self) -> List[ShardStatus]:
        total_shards = int((self.status_folder / "total_shards.txt").read_text())
        shard_sizes = self._get_shard_sizes()

        if total_shards != len(shard_sizes):
            raise RuntimeError(
                "There is a mismatch between the reported number of shards on disk and the number of shards we need."
            )

        shard_statuses = []
        for i in range(total_shards):
            shard_status_file = self.status_folder / f"{i}.txt"
            if not shard_status_file.exists():
                # does not exist yet
                continue

            shard_status = shard_status_file.read_text()
            if shard_status == "done":
                shard_statuses.append(ShardStatus.DONE)
            elif shard_status.startswith("failed"):
                shard_statuses.append(ShardStatus.FAIL)
            elif shard_status.startswith("in progress"):
                started_at_ts = int(shard_status.splitlines()[1])
                started_at_dt = datetime.utcfromtimestamp(started_at_ts)
                current_time = datetime.utcnow()
                diff_in_seconds = (current_time - started_at_dt).total_seconds()
                diff_in_hours = diff_in_seconds / 3600
                did_fail = diff_in_hours < JOB_HOURS_FAILURE_THRESHOLD
                shard_statuses.append(ShardStatus.FAIL if did_fail else ShardStatus.IN_PROGRESS)

        return shard_statuses

    def get_bash_scripts_for_failed_shards(self) -> List[Path]:
        shard_statuses = self.get_shard_statuses()
        failed_shard_script_paths = []
        for i, shard_status in enumerate(shard_statuses):
            if shard_status == ShardStatus.FAIL:
                tmp_file = self.dataset_folder / f"tmp_slurm_{self.model_config_checksum}_{i}.sh"
                assert tmp_file.exists()
                failed_shard_script_paths.append(tmp_file)
        return failed_shard_script_paths

    def get_total_shards(self) -> int:
        return int((self.status_folder / "total_shards.txt").read_text())

    def get_bash_scripts_for_all_shards(self) -> List[Path]:
        all_shard_script_paths = []
        for i in range(self.get_total_shards()):
            tmp_file = self.dataset_folder / f"tmp_slurm_{self.model_config_checksum}_{i}.sh"
            if tmp_file.exists():
                all_shard_script_paths.append(tmp_file)
        return all_shard_script_paths
    
    def write_shard_runner_scripts_and_embedding_info_csv(self, conda_env_name: str, slurm_partition: str) -> List[Path]:
        if self.embeddings_info_file.is_file():
            raise RuntimeError(
                f"The embeddings info file already exists, delete this file: {self.embeddings_info_file} if the intent is to regenerate the embeddings from scratch."
            )
        
        row_data = []
        slurm_files = []
        shard_idx = -1
        for sample_idx in range(0, self.num_total_samples, self.max_samples_per_shard):
            # shards are 0-indexed
            shard_idx += 1

            # write the script that will populate this shard
            json_str = json.dumps(self.model_config, indent=4)
            json_comment = "\n".join(f"# {x}" for x in json_str.splitlines())
            script_contents = SLURM_JOB_BASE.format(
                slurm_partition=slurm_partition,
                hash=self.model_config_checksum,
                dataset_folder_name=self.dataset_name,
                dataset_shard=shard_idx,
                conda_env_name=conda_env_name,
                model_config_str=json_comment,
            ).strip()

            tmp_file = self.dataset_folder / f"tmp_slurm_{self.model_config_checksum}_{shard_idx}.sh"
            tmp_file.write_text(script_contents)
            slurm_files.append(tmp_file)

            # write a row for all the samples in this shard
            t_sample_info: NamedTuple
            for i, t_sample_info in enumerate(
                self.dataset_info_df.iloc[
                    sample_idx : sample_idx + self.max_samples_per_shard
                ].itertuples()
            ):
                sample_idx_in_dataset = i + sample_idx
                sample_info = t_sample_info._asdict()

                # use the offset sample if it exists
                audio_filepath = get_audio_file_path_from_sample_info(sample_info)

                row = {
                    "zarr_file_path": str(self.zarr_file_path.absolute()),
                    "zarr_idx": sample_idx_in_dataset,
                    "model_name": self.model_name,
                    "model_config_checksum": self.model_config_checksum,
                    "dataset_name": self.dataset_name,
                    "dataset_shard": shard_idx,
                    "audio_file_path": audio_filepath,
                    # retain full original sample information
                    "details": sample_info,
                }
                row_data.append(row)

        embedding_info_df = pd.json_normalize(row_data)
        embedding_info_df.to_csv(self.embeddings_info_file)

        return slurm_files

    @classmethod
    def load_from_dataset_folder_and_checksum(cls, dataset_folder: Path, model_config_checksum: str) -> "DatasetEmbeddingInformation":
        dataset_folder_name = dataset_folder.parts[-1]
        shard_prefix = f"{dataset_folder_name}_{model_config_checksum}"
        model_config_path = dataset_folder / (shard_prefix + ".json")
        model_config = json.loads(model_config_path.read_text())
        return DatasetEmbeddingInformation(
            dataset_folder=dataset_folder,
            model_config=model_config,
            # do not use this parameter anymore, the scripts to extract shards should exist
            # on disk already
            max_samples_per_shard=1
        )


def extract_shard(
    dataset_folder_name: str,
    dataset_shard: int,
    model_config_checksum: str,
    root_dir: Optional[Path] = OUTPUT_DIR,
) -> Tuple[zarr, np.ndarray]:
    dataset_folder = root_dir / dataset_folder_name
    embedding_info = DatasetEmbeddingInformation.load_from_dataset_folder_and_checksum(dataset_folder, model_config_checksum)

    # load information for this model embedding
    model_config = embedding_info.model_config
    info_file = embedding_info.embeddings_info_file
    status_folder = embedding_info.status_folder

    # overwrites existing text
    started_at = int(time.time())
    shard_status_path = status_folder / (f"{dataset_shard}.txt")
    shard_status_path.write_text(f"in progress\n{started_at}")

    # read the embeddings information, will use this to determine if we need
    # to extract a specific index yet
    embeddings_info_df = pd.read_csv(info_file)

    # load model
    model_type = Model[model_config["model_type"]]
    try:
        processor, model = load_musicgen_model(model_type)
    except ValueError:
        # not a musicgen model, later functions will handle
        processor, model = None, None

    zarr_file = embedding_info.load_zarr_file()
    if not zarr_file:
        raise RuntimeError(f"Embeddings file for {dataset_folder_name} ({model_config_checksum}) did not exist.")

    written_idx = []
    try:
        t_sample_info: NamedTuple
        for t_sample_info in embeddings_info_df.itertuples():
            sample_info = t_sample_info._asdict()

            if int(sample_info["dataset_shard"]) != dataset_shard:
                # only convert the files required by this shard
                continue

            sample_idx = int(sample_info["zarr_idx"])
            written_idx.append(sample_idx)
            audio_file_path = sample_info["audio_file_path"]

            if not np.any(zarr_file[sample_idx]):
                # all 0s in this array slice, hasn't yet been written, do not overwrite already written
                embedding_vec = get_embedding_from_model_using_config(
                    dataset_folder / audio_file_path, model_config, processor, model
                )

                zarr_file[sample_idx] = embedding_vec

                print(
                    f"Shard: {dataset_shard}: Finished extracting idx: {sample_idx}, "
                    f"file: {audio_file_path}, for dataset: {dataset_folder_name}, "
                    f"with model config checksum: {model_config_checksum} ({model_type})"
                )
                # from zarr docs: "files are automatically closed whenever an array is modified."

    except Exception as e:
        shard_status_path.write_text(f"failed. Error: {e}")
        raise e

    # overwrites existing text
    shard_status_path.write_text("done")

    return zarr_file, np.array(written_idx)


def get_embedding_from_model_using_config(
    audio_file: Path,
    model_config: Dict[str, Any],
    processor: AutoProcessor = None,
    model: MusicgenForConditionalGeneration = None,
) -> np.ndarray:
    model_type = Model[model_config["model_type"]]
    minimum_duration = model_config["minimum_duration_in_sec"]

    audio, sr = torchaudio.load(audio_file)
    duration = audio.shape[-1] / sr

    if duration < minimum_duration:
        raise ValueError(
            f"Audio file at location: {minimum_duration} is not long enough."
        )

    embedding = audio_file_to_embedding_np_array(
        audio_file,
        model_type,
        processor,
        model,
        extract_from_layer=model_config.get("extract_from_layer", None),
        # decoder hidden states defaults to True
        decoder_hidden_states=model_config.get("decoder_hidden_states", True),
        # meanpool defaults to True
        meanpool=model_config.get("meanpool", True),
    )
    return embedding


def get_scripts_to_extract_embeddings_for_dataset_with_model(
    dataset_folder: Path,
    model_config: Dict[str, Any],
    conda_env_name: str,
    slurm_partition: str,
    max_samples_per_shard: int = 300,
) -> List[Path]:
    dataset_coordinator = DatasetEmbeddingInformation(dataset_folder, model_config, max_samples_per_shard)
    embeddings_array = dataset_coordinator.get_or_create_zarr_file()
    
    # check if there are any shards that have failed
    failed_shards = dataset_coordinator.get_bash_scripts_for_failed_shards()
    if failed_shards:
        failed_path_text = ", ".join([str(x.absolute()) for x in failed_shards])
        raise RuntimeError(f"There are failed shards. Consider inspecting and re-running the following scripts: {failed_path_text}")

    slurm_files = dataset_coordinator.write_shard_runner_scripts_and_embedding_info_csv(
        conda_env_name, slurm_partition
    )

    return slurm_files


def extract_embeddings_for_dataset_with_model(
    dataset_folder: Path,
    model_config: Dict[str, Any],
    conda_env_name: str,
    slurm_partition: str,
    max_samples_per_shard: int = 300,
) -> List[Path]:
    dataset_coordinator = DatasetEmbeddingInformation(dataset_folder, model_config, max_samples_per_shard)
    
    # create the zarr file to hold the embeddings
    embeddings_array = dataset_coordinator.get_or_create_zarr_file()
    
    # write the scripts that we can use to run to extract a portion (shard) of all embeddings
    slurm_files = dataset_coordinator.write_shard_runner_scripts_and_embedding_info_csv(
        conda_env_name, slurm_partition
    )

    # enqueue the jobs to extract the shards
    for tmp_file in slurm_files:
        subprocess.run(f"chmod u+x {tmp_file.absolute()}", shell=True)
        subprocess.run(f"sbatch {tmp_file.absolute()}", shell=True)

    return slurm_files

def has_no_shard_scripts(dataset_folder: Path, model_config: Dict[str, Any], max_samples_per_shard: int) -> bool:
    dataset_coordinator = DatasetEmbeddingInformation(dataset_folder, model_config, max_samples_per_shard)
    # no shards have been written yet, so it does have unfinished jobs by definition
    return (
        dataset_coordinator.get_total_shards() != 0 and len(dataset_coordinator.get_bash_scripts_for_all_shards()) == 0
    )

def get_failed_jobs(dataset_folder: Path, model_config: Dict[str, Any], max_samples_per_shard: int) -> List[Path]:
    dataset_coordinator = DatasetEmbeddingInformation(dataset_folder, model_config, max_samples_per_shard)
    return (
        # has shards that have FAILED status
        dataset_coordinator.get_bash_scripts_for_failed_shards()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)

    models = config['models']
    concepts = config['concepts']
    settings = config['settings']

    slurm_partition = settings['slurm_partition']
    conda_env_name = settings['conda_env_name']
    
    with use_770_permissions():
        for concept in concepts:
            dataset_folder = OUTPUT_DIR / concept
            
            for model_name in models:
                model_config = {
                    "model_name": model_name,
                    "model_type": model_name,
                    "minimum_duration_in_sec": settings['minimum_duration_in_sec']
                }

                if model_name == "JUKEBOX":
                    model_config["decoder_hidden_states"] = False

                if has_no_shard_scripts(dataset_folder, model_config, settings['max_samples_per_shard']):
                    # - no scripts at all (never started)
                    print(f"Extracting embeddings for {concept} using {model_name}")
                    shard_scripts = extract_embeddings_for_dataset_with_model(
                        dataset_folder, 
                        model_config, 
                        conda_env_name,
                        slurm_partition,
                        max_samples_per_shard=settings['max_samples_per_shard']
                    )
                else:
                    failed_jobs = get_failed_jobs(dataset_folder, model_config, settings['max_samples_per_shard'])
                    if failed_jobs:
                        # - all or some scripts have failed
                        scripts_str = ', '.join([x.name for x in failed_jobs])
                        print(f"There are failed jobs for some shards. Try re-running these scripts: {scripts_str}.")
                    else:
                        # - all scripts are in progress
                        print(f"{concept} - {model_name} ({compute_checksum(model_config)}) is done with no errors.")


if __name__ == "__main__":
    main()