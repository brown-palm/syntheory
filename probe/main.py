import os
import traceback
import argparse
from pathlib import Path
from json import loads

import zarr
import wandb

from config import OUTPUT_DIR
from probe.probes import ProbeExperiment, ProbeExperimentConfig
from probe.probe_config import CONCEPT_LABELS


def get_all_embedding_exports(
    concept_name: str,
    base_path_parent: Path = OUTPUT_DIR,
):
    """Each zarr file represents embeddings extracted from a specific model.

    The properties of these models are
    - model type [Jukebox | Musicgen]
    - model size [S | M | L]
    - model layer [1 ... n]

    """
    base_path = base_path_parent / concept_name
    dataset_info = base_path / "info.csv"
    zarr_files = list(base_path.glob("*.zarr"))
    model_embeddings_infos = []
    for z in zarr_files:
        f_name = z.parts[-1]

        if not f_name.startswith(concept_name):
            # expect embeddings arrays to be prefixed with the concept name
            continue

        # get the settings associated with the model that produced these embeddings
        model_hash = f_name.split(".zarr")[0].split("_")[-1]
        model_settings_path = base_path / (concept_name + "_" + model_hash + ".json")
        model_settings = loads(model_settings_path.read_text())

        emb = zarr.open(z)

        model_name = model_settings["model_name"]
        if model_name == "JUKEBOX":
            model_size = "L"
            model_type = "JUKEBOX"
        elif "MUSICGEN_DECODER" in model_name:
            model_size = model_name.split("_LM_")[-1]
            model_type = "MUSICGEN_DECODER"
        elif "MUSICGEN_AUDIO_ENCODER" in model_name:
            model_size = "L"
            model_type = "MUSICGEN_AUDIO_ENCODER"
        else:
            # handcrafted features, default to L
            model_size = "L"
            model_type = model_settings["model_name"]

        exp_info = {
            "zarr_filepath": z,
            "dataset_labels_path": dataset_info,
            "embeddings_dataset_shape": emb.shape,
            "model_hash": model_hash,
            "model_settings": model_settings,
            "model_size": model_size,
            "model_type": model_type,
        }
        model_embeddings_infos.append(exp_info)

    return model_embeddings_infos


def _set_attr_if_exists(probe_config, hparams, attr_name, default=None):
    x = getattr(probe_config, attr_name, default)
    if x is not None:
        hparams[attr_name] = x


def _is_equal_model_types(mt_a: str, mt_b: str) -> bool:
    # in some configs we neglected the "LM" suffix. Consider
    # "MUSICGEN_DECODER_LM" to be equal to "MUSICGEN_DECODER"
    if mt_a.startswith("MUSICGEN_DECODER") and mt_b.startswith("MUSICGEN_DECODER"):
        return mt_a.split("_LM")[0] == mt_b.split("_LM")[0]

    return mt_a == mt_b


def start(
    use_wandb: bool = True, random_seed: int = 0, base_path_parent: Path = OUTPUT_DIR
) -> ProbeExperiment:
    # the sweep configuration will exist in this object, use it to get the
    # dataset and experiment configuration
    probe_config = wandb.config

    # model type: [ JUKEBOX | MUSICGEN_DECODER | MUSICGEN_AUDIO_ENCODER | MFCC | CHROMA | MELSPEC | HANDCRAFT ]
    model_type = probe_config.model_type
    # model size: [S | M | L]
    model_size = probe_config.model_size
    # model layer: [0, ... 71]
    model_layer = probe_config.model_layer
    # concept: [notes, tempos, time_signatures, etc. ] + a specific label
    concept = probe_config.concept

    num_classes = getattr(probe_config, "num_classes", None)

    # set hyperparameters
    hparams = {}
    _set_attr_if_exists(probe_config, hparams, "data_standardization")
    _set_attr_if_exists(probe_config, hparams, "batch_size")
    _set_attr_if_exists(probe_config, hparams, "learning_rate")
    _set_attr_if_exists(probe_config, hparams, "dropout_p")
    _set_attr_if_exists(probe_config, hparams, "l2_weight_decay")
    _set_attr_if_exists(probe_config, hparams, "hidden_layer_sizes", [512])

    # get the concept label that is given by parent concept and the target we wish to probe
    dataset_settings = CONCEPT_LABELS[concept][0]
    _num_classes, label_column_name = dataset_settings

    # allow override of number of classes if given in config directly
    num_classes = num_classes or _num_classes
    concept_parent_name = concept

    # look up the correct location for the embeddings given experiment config
    exp_info = None
    for e in get_all_embedding_exports(concept_parent_name, base_path_parent):
        if (
            e["model_size"] == model_size
            and _is_equal_model_types(e["model_type"], model_type)
            # shape is (all samples, layer, embedding dimension)
            and e["embeddings_dataset_shape"][1] > model_layer
        ):
            exp_info = e

    if exp_info is None:
        print(
            "There is no dataset corresponding to the config specified in the sweep."
        )
        # nothing to do if a configuration for this does not exist. This might happen if
        # we try to run a probe for S or M models of JUKEBOX or for larger layers of
        # MUSICGEN that do not exist. Return nothing instead of throwing an error to treat it
        # as a noop instead of a job failure.
        return

    is_regression = concept_parent_name == "tempos"
    num_outputs = 1 if is_regression else num_classes
    # can be: 'multiclass' or 'regression'
    output_type = "regression" if is_regression else "multiclass"

    wandb.config["num_outputs"] = num_outputs
    wandb.config["output_type"] = output_type
    wandb.config["label_column_name"] = label_column_name

    cfg = ProbeExperimentConfig(
        dataset_embeddings_label_column_name=label_column_name,
        dataset=concept,
        num_outputs=num_outputs,
        model_hash=f"{model_type}-{model_size}-{model_layer}",
        max_num_epochs=100,
        **hparams,
        seed=random_seed,
        # intervals and chord progressions are too large to fit in ram, for all others
        # we can load into memory, but for those we need to load off disk as we train.
        load_embeddings_in_memory=(
            concept_parent_name not in ("intervals", "chord_progressions")
        ),
    )

    exp = ProbeExperiment(
        cfg,
        summarize_frequency=100,
        use_wandb=use_wandb,
    )

    emb_in_mem = cfg["load_embeddings_in_memory"]

    # --- TRAIN PROBE ---
    exp.load_data(
        dataset_labels_filepath=exp_info["dataset_labels_path"],
        dataset_label_column_name=label_column_name,
        embeddings_zarr_filepath=exp_info["zarr_filepath"],
        output_type=output_type,
        model_layer=model_layer,
    )
    exp.train()

    return exp


def wrapped_train() -> None:  # pragma: no cover
    try:
        wandb.init()
        start()
    except Exception as e:
        wandb.log({"error": str(e), "traceback": traceback.format_exc()})
        raise e


if __name__ == "__main__":  # pragma: no cover
    """From within a wandb sweep, setup a simple probe to train on the dataset task. 

    This file should be called by a SLURM job that is initialized by run_probes.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    args = parser.parse_args()

    # interesting bug, see: https://github.com/wandb/wandb/issues/5272#issuecomment-1881950880
    # only applies when doing hyperparameter sweeps with slurm.
    os.environ["WANDB_DISABLE_SERVICE"] = "True"

    # get the wandb sweep ID and launch the agent to perform some runs
    wandb.agent(
        args.sweep_id, project=args.wandb_project, function=wrapped_train, count=1
    )
