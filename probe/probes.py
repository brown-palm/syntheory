"""
Adapted from: https://github.com/p-lambda/jukemir/blob/main/jukemir/probe/__init__.py
"""
import json
import logging
import math
import pickle
import random
import tempfile
from pathlib import Path
from os import environ as os_env
from typing import List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
import sklearn
import zarr
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score

from sklearn.metrics import (
    average_precision_score,
    r2_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from probe.probe_config import ProbeExperimentConfig


# -- handle jukemir cache --
if "JUKEMIR_CACHE_DIR" in os_env:
    CACHE_DIR = Path(os_env["JUKEMIR_CACHE_DIR"])
else:
    CACHE_DIR = Path(Path.home(), ".jukemir")
CACHE_DIR = CACHE_DIR.resolve()
CACHE_PROBES_DIR = Path(CACHE_DIR, "probes")


class SimpleMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_layer_sizes: List[int],
        num_outputs: int,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        d = num_features

        self.num_layers = len(hidden_layer_sizes)
        for i, ld in enumerate(hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld

        self.output = nn.Linear(d, num_outputs)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.output(x)


class ProbeExperiment:
    OHE_SUFFIX = "_ohe"

    def __init__(
        self,
        cfg: ProbeExperimentConfig,
        pretrained_scaler=None,
        pretrained_probe=None,
        summarize_frequency: int = 20,
        use_wandb: bool = True,
    ) -> None:
        if not cfg["early_stopping"] and cfg["max_num_epochs"] is None:
            raise ValueError("No termination criteria specified")

        self.cfg = cfg
        self.scaler = pretrained_scaler
        self.probe = pretrained_probe
        self.label_column = cfg["dataset_embeddings_label_column_name"]
        self.use_wandb = use_wandb

        # print a summary ever n steps
        self.summarize_frequency = summarize_frequency
        self.random_seed = cfg["seed"]

        # Set seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            np.random.seed(self.random_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.probe is not None:
            self.probe.to(self.device)

    def _get_map_for_col_name(self, col_name: str, df: pd.DataFrame):
        return {
            v: k
            for k, v in dict(enumerate(sorted(set(df[col_name].to_list())))).items()
        }

    def get_train_test_valid_split_from_pandas_df(
        self, df: pd.DataFrame, train_size: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_shuffled = df.sample(frac=1, random_state=self.random_seed)

        # k train, (1-k)/2 test, (1-k)/2 valid
        train_df, non_train_df = train_test_split(
            df_shuffled, train_size=train_size, random_state=self.random_seed
        )
        valid_df, test_df = train_test_split(
            non_train_df, test_size=0.5, random_state=self.random_seed
        )

        # Optionally, reset the index of the split DataFrames
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, test_df, valid_df

    def format_dataset_labels(
        self, output_type: str, dataset_label_column_name: str, dataset_labels
    ):
        if output_type == "regression":
            # normalize the outputs in a regression task to be in [0, 1]
            scaler = MinMaxScaler()
            dataset_labels[dataset_label_column_name] = scaler.fit_transform(
                dataset_labels[[dataset_label_column_name]]
            )
            dataset_labels[dataset_label_column_name] = dataset_labels[
                dataset_label_column_name
            ].astype("float32")
        elif output_type == "multiclass":
            # we are doing multiclass classification, column values to categoricals
            dataset_labels[dataset_label_column_name + "_encoded"] = pd.Categorical(
                dataset_labels[dataset_label_column_name]
            ).codes
        else:
            raise ValueError(f"Unsupported output type, got: {output_type}")

        label_map = self._get_map_for_col_name(
            dataset_label_column_name, dataset_labels
        )

        dataset_labels[dataset_label_column_name + "_encoded"] = dataset_labels[
            dataset_label_column_name
        ].apply(lambda x: label_map[x])

        # add one hot encoded labels to the dataframe
        dataset_labels[self.label_column + self.OHE_SUFFIX] = (
            pd.get_dummies(dataset_labels[self.label_column])
            .astype(np.float32)
            .values.tolist()
        )
        return dataset_labels

    def load_data(
        self,
        dataset_labels_filepath,
        dataset_label_column_name: str,
        embeddings_zarr_filepath,
        output_type: str,
        model_layer: int = 0,
    ) -> None:
        self.dataset_labels_filepath = dataset_labels_filepath

        # load the dataset and encode categorical targets
        dataset_labels = pd.read_csv(dataset_labels_filepath)
        dataset_labels = self.format_dataset_labels(
            output_type, dataset_label_column_name, dataset_labels
        )

        # add the zarr idx if not listed explicitly
        if "zarr_idx" not in set(dataset_labels.columns):
            dataset_labels["zarr_idx"] = np.arange(dataset_labels.shape[0])

        # get test / train / validation split
        if self.cfg["dataset"] == "tempos" and output_type == "regression":
            # order is important, we want to test the out of domain extrapolation power of
            # the embeddings.
            dataset_labels = dataset_labels.sort_values(by=["bpm"])
            ends = 0.15
            dataset_size = dataset_labels.shape[0]
            lb_idx = int(dataset_size * (ends))
            ub_idx = int(dataset_size * (1 - ends))

            # train on BPMs within a specific middle range
            train_df = dataset_labels.iloc[lb_idx:ub_idx]
            hold_out_df = pd.concat(
                [dataset_labels.iloc[:lb_idx], dataset_labels.iloc[ub_idx:]],
                axis=0,
            )
            hold_out_shuffled = hold_out_df.sample(
                frac=1, random_state=self.random_seed
            )
            test_df, valid_df = train_test_split(
                hold_out_shuffled, train_size=0.5, random_state=self.random_seed
            )
            # our train/test/valid splits are still 0.7, 0.15, 0.15.
        else:
            train_df, test_df, valid_df = (
                self.get_train_test_valid_split_from_pandas_df(dataset_labels)
            )

        # load the zarr, read only mode
        data = zarr.open(embeddings_zarr_filepath, mode="r")

        self.is_foundation_model_layers = len(data.shape) == 3
        self.embeddings = data

        # Organize data
        self.split_to_uids = {"train": [], "valid": [], "test": []}
        self.split_to_X = {}
        self.split_to_y = {}

        for split in (("train", train_df), ("test", test_df), ("valid", valid_df)):
            split_name, data_df = split
            selector = data_df["zarr_idx"].to_numpy()

            if self.cfg["load_embeddings_in_memory"]:
                # load embeddings in memory for faster training.
                # they will be in an array of dimension: (n, layer_num, k)
                # where
                #   - n is the sample index
                #   - layer_num is the layer from which the embedding was extracted
                #   - k is the dimensionality of the embedding
                if self.is_foundation_model_layers:
                    X = np.array(data[selector][:, model_layer, :], dtype=np.float32)
                else:
                    # handcrafted features
                    X = np.array(data[selector], dtype=np.float32)
            else:
                if self.is_foundation_model_layers:
                    X = (selector, model_layer)
                else:
                    # handcrafted features
                    X = (selector,)

            # label
            if output_type == "regression":
                # assume y is some numerical value, may have been scaled
                y = data_df[self.label_column].to_list()
            else:
                # encode it using ohe
                y = np.array(
                    data_df[self.label_column + self.OHE_SUFFIX].tolist()
                ).astype(np.float32)

            # tuple of numpy_array, integer of model layer
            self.split_to_X[split_name] = X
            self.split_to_y[split_name] = y

        # multi-class, multi-label, regression
        self.output_type = output_type

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output_type = self.output_type
        if output_type == "multiclass":
            loss = F.cross_entropy(logits, y, reduction="mean")
        elif output_type == "regression":
            if (
                logits.shape[0] == y.shape[0]
                and len(logits.shape) == 2
                and len(y.shape) == 1
            ):
                logits = logits.squeeze(1)
            elif logits.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch between model output and targets. Logits: {logits.shape}, y: {y.shape}"
                )
            loss = F.mse_loss(logits.float(), y.float(), reduction="mean")
        else:
            raise NotImplementedError()

        return loss

    @staticmethod
    def destructure(t: Any) -> Tuple[Any, Any]:
        if isinstance(t, np.ndarray):
            return t, -1
        try:
            x, y = t
            return x, y
        except ValueError:
            return t[0], -1

    def train(self) -> None:
        # extract a single embedding to get its dimension
        if self.cfg["load_embeddings_in_memory"]:
            single_X = self.split_to_X["train"][0, :]
            embedding_dimension = single_X.shape[0]
        else:
            X_train, model_layer = self.destructure(self.split_to_X["train"])
            if model_layer != -1:
                single_X = self.embeddings[X_train[:1]][:, model_layer, :]
            else:
                single_X = self.embeddings[X_train[:1]]

            embedding_dimension = single_X.shape[1]

        self.probe = SimpleMLP(
            embedding_dimension,
            self.cfg["hidden_layer_sizes"],
            num_outputs=self.cfg["num_outputs"],
            dropout_p=self.cfg["dropout_p"],
        )
        self.probe.to(self.device)
        self.probe.train()

        # Create optimizer
        optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=(
                0
                if self.cfg["l2_weight_decay"] is None
                else self.cfg["l2_weight_decay"]
            ),
        )

        # Retrieve dataset, must have run load_data beforehand
        X_train, model_layer = self.destructure(self.split_to_X["train"])

        y_train = np.array(self.split_to_y["train"])

        # Fit scaler
        self.scaler = StandardScaler(
            with_mean=self.cfg["data_standardization"],
            with_std=self.cfg["data_standardization"],
        )
        # not feasible to fit all at once, see in training we use partial_fit which is
        # online computation of mean/std for batches.
        # self.scaler.fit(X_train)
        self.metrics_for_graph = []

        # Train model
        step = 0
        early_stopping_best_score = float("-inf")
        early_stopping_boredom = 0
        early_stopping_state_dict = None

        while True:
            # Check if exceeded max num epochs
            epoch = (step * self.cfg["batch_size"]) / X_train.shape[0]
            if (
                self.cfg["max_num_epochs"] is not None
                and epoch > self.cfg["max_num_epochs"]
            ):
                break

            # Evaluate for early stopping
            if (
                self.cfg["early_stopping"]
                and step % self.cfg["early_stopping_eval_frequency"] == 0
            ):
                if early_stopping_boredom >= self.cfg["early_stopping_boredom"]:
                    if early_stopping_state_dict is not None:
                        self.probe.load_state_dict(early_stopping_state_dict)
                    break

                with torch.no_grad():
                    self.probe.eval()
                    metrics = self.eval("valid")

                    if self.cfg["early_stopping_metric"].startswith("-"):
                        score = -1 * metrics[self.cfg["early_stopping_metric"][1:]]
                    else:
                        score = metrics[self.cfg["early_stopping_metric"]]

                    self.probe.train()

                    logging.info(f"eval,{step},{score}")

                    metrics.update(
                        {
                            "epoch": epoch,
                            "early_stopping_score": score,
                            "early_stopping_best_score": early_stopping_best_score,
                            "early_stopping_boredom": early_stopping_boredom,
                        }
                    )
                    if self.use_wandb:
                        wandb.log(metrics, step=step)

                    if math.isnan(score):
                        raise Exception("NaN score")

                    if score > early_stopping_best_score:
                        early_stopping_best_score = score
                        early_stopping_boredom = 0
                        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                            torch.save(self.probe.state_dict(), f.name)
                            early_stopping_state_dict = torch.load(f.name)
                    else:
                        early_stopping_boredom += 1

            # Create batch
            idxs = random.sample(
                list(range(X_train.shape[0])),
                min(self.cfg["batch_size"], X_train.shape[0]),
            )

            if self.cfg["load_embeddings_in_memory"]:
                # load embeddings directly, since they are stored in training splits
                X = X_train[idxs, :]
            else:
                # indexes are stored in splits, retrieve them from disk
                if model_layer != -1:
                    X = self.embeddings[X_train[idxs], model_layer, :]
                else:
                    X = self.embeddings[X_train[idxs], :]

            y = y_train[idxs]

            self.scaler.partial_fit(X)
            X = self.scaler.transform(X)
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

            y = torch.tensor(y, device=self.device)

            # Update
            optimizer.zero_grad()
            loss = self.compute_loss(self.probe(X), y)
            loss.backward()
            optimizer.step()
            step += 1

            # Summarize
            if step % self.summarize_frequency == 0:
                loss = loss.item()
                logging.debug(f"train,{step},{loss}")

                if self.use_wandb:
                    wandb.log({"train_loss": loss}, step=step)

    def eval_logits(self, X: torch.Tensor) -> torch.Tensor:
        try:
            X = self.scaler.transform(X)
        except sklearn.exceptions.NotFittedError:
            self.scaler.partial_fit(X)
            X = self.scaler.transform(X)

        with torch.no_grad():
            self.probe.eval()
            logits = []
            for i in range(0, X.shape[0], self.cfg["batch_size"]):
                X_batch = torch.tensor(
                    X[i : i + self.cfg["batch_size"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                logits.append(self.probe(X_batch))
            logits = torch.cat(logits, dim=0)

        return logits

    def eval(
        self, uids_or_split_name: str, X=None, y=None, with_confusion_matrix: bool = False
    ):
        assert isinstance(uids_or_split_name, str)
        split_name = uids_or_split_name
        uids = self.split_to_uids[split_name]
        if self.cfg["load_embeddings_in_memory"]:
            X = self.split_to_X[split_name]
        else:
            X_idxs, model_layer = self.destructure(self.split_to_X[split_name])
            if model_layer != -1:
                X = self.embeddings[X_idxs, model_layer, :]
            else:
                X = self.embeddings[X_idxs, :]

        y = self.split_to_y[split_name]

        if self.output_type == "regression":
            # [bs] --> [bs, 1]
            y = np.array(y, dtype=np.float32)
            y = np.expand_dims(y, axis=-1)

        metrics = {}
        primary_metric_name = None

        # Compute logits / task-specific loss
        with torch.no_grad():
            self.probe.eval()
            logits = self.eval_logits(X)
            y_tensor = torch.tensor(y, device=self.device)
            metrics["loss"] = self.compute_loss(logits, y_tensor).item()
            logits = logits.cpu().numpy()

        # Compute evaluation metrics
        if self.output_type == "multiclass":
            primary_metric_name = "accuracy"
            y_preds = np.argmax(logits, axis=1)
            y = np.argmax(y, axis=1)
            y_correct = y_preds == y

            metrics["accuracy"] = y_correct.astype(np.float32).mean()
            metrics["primary_eval_metric"] = metrics["accuracy"]
            metrics["f1"] = multiclass_f1_score(
                torch.tensor(y_preds),
                torch.tensor(y),
                num_classes=self.cfg["num_outputs"],
            ).item()

            if with_confusion_matrix:
                cm = confusion_matrix(y, y_preds)
                metrics["confusion_matrix"] = cm

        elif self.output_type == "regression":
            # assuming only 1 scalar output
            primary_metric_name = "r2"

            # y and logits are normalized
            metrics["r2"] = r2_score(y, logits)
            metrics["primary_eval_metric"] = metrics["r2"]
        else:
            raise NotImplementedError()

        # Convert to simple Python types
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, np.generic)):
                metrics[k] = v.tolist()

        assert "primary" not in metrics
        metrics["primary"] = metrics[primary_metric_name]
        return metrics

    def save(self, root_dir: Optional[Union[str, Path]] = None) -> Tuple[str, Path]:
        if root_dir is None:
            root_dir = CACHE_PROBES_DIR

        uid = self.cfg.uid()
        model_dir = Path(root_dir, self.cfg["dataset"], uid)
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(model_dir, "cfg.json"), "w") as f:
            f.write(json.dumps(self.cfg, indent=2, sort_keys=True))

        with open(Path(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        torch.save(self.probe.state_dict(), Path(model_dir, "probe.pt"))
        self.save_metrics(root_dir)

        return uid, root_dir

    def delete(
        self, root_dir: Optional[str] = None, delete_metrics_and_config: bool = False
    ) -> None:
        if root_dir is None:
            root_dir = CACHE_PROBES_DIR

        uid = self.cfg.uid()
        model_dir = Path(root_dir, self.cfg["dataset"], uid)
        model_dir.mkdir(parents=True, exist_ok=True)

        if delete_metrics_and_config:
            Path(model_dir, "metrics.json").unlink(missing_ok=True)
            Path(model_dir, "cfg.json").unlink(missing_ok=True)

        Path(model_dir, "scaler.pkl").unlink(missing_ok=True)
        Path(model_dir, "probe.pt").unlink(missing_ok=True)

    def save_metrics(self, root_dir: Optional[str] = None) -> None:
        with open(Path(root_dir, "metrics.json"), "w") as f:
            f.write(json.dumps(self.eval("valid"), indent=2, sort_keys=True))

    @classmethod
    def load(cls, uid: str, root_dir=CACHE_PROBES_DIR, **kwargs) -> "ProbeExperiment":
        # load the model state
        model_dir = [d for d in Path(root_dir).rglob(f"{uid}*") if d.is_dir()]
        if len(model_dir) < 1:
            raise ValueError("Could not find model directory")

        model_dir = model_dir[0]

        with open(Path(model_dir, "cfg.json"), "r") as f:
            cfg = ProbeExperimentConfig(json.load(f))

        with open(Path(model_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        state_dict = torch.load(
            Path(model_dir, "probe.pt"),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        if len(cfg["hidden_layer_sizes"]) > 0:
            input_layer = "hidden_0.weight"
        else:
            input_layer = "output.weight"

        # construct the probe given the config
        probe = SimpleMLP(
            state_dict[input_layer].shape[1],
            cfg["hidden_layer_sizes"],
            num_outputs=cfg["num_outputs"],
            dropout_p=cfg["dropout_p"],
        )
        probe.load_state_dict(state_dict)

        # return instance of ProbeExperiment, loaded from persisted config + model state
        return cls(cfg, pretrained_scaler=scaler, pretrained_probe=probe, **kwargs)
