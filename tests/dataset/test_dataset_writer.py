from typing import Iterator, List
from pathlib import Path
import tempfile
import uuid

import pytest
import pandas as pd

from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription


def _get_row_iterator() -> Iterator[DatasetRowDescription]:
    for i in range(100):
        yield (i, {"my_id": str(uuid.uuid4())})


def _row_processor(
    parent_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_config = row
    return [
        (
            row_idx,
            {
                "my_id": row_config["my_id"][::-1],
                "something_else": 1,
                "parent_path": parent_path,
            },
        )
    ]


def _row_processor_that_throws_exception(
    parent_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_config = row
    if row_idx > 3:
        raise RuntimeError("Row Processor Error For Test")

    return [
        (
            row_idx,
            {
                "my_id": row_config["my_id"][::-1],
                "something_else": 1,
                "parent_path": parent_path,
            },
        )
    ]


def test_dataset_writer() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_writer_instance = DatasetWriter(
            "test", Path(tmp_dir), _get_row_iterator(), _row_processor, max_processes=2
        )
        dataset_df = dataset_writer_instance.create_dataset()
        assert isinstance(dataset_df, pd.DataFrame)
        results = dataset_df.to_dict("records")
        row_id = -1
        # check that it is sorted by this, and the objects are structured as we expect
        for result in results:
            assert row_id < result["row_id"]
            row_id = result["row_id"]
            assert result["something_else"] == 1
            assert len(result["my_id"]) == 36  # length of uuid4

        # can load the dataset from disk if it already exists
        df = dataset_writer_instance.get_dataset_as_pandas_dataframe()
        assert df.shape == dataset_df.shape

        # if the dataset already exists though, we should not be able to mutate / append to
        # the folders and files that contain it
        with pytest.raises(RuntimeError) as exc:
            dataset_writer_instance.create_dataset()
            assert str(exc.value).startswith(
                "A dataset folder at this location already exists!"
            )


def test_dataset_writer_path_type_exception() -> None:
    with pytest.raises(ValueError) as exc:
        dataset_writer_instance = DatasetWriter(
            "test", "./tmp_dir", _get_row_iterator(), _row_processor, max_processes=1
        )
        assert (
            str(exc.value)
            == "save_to_parent_directory must be a Path object. It was given as type: <class 'str'>"
        )


def test_dataset_writer_row_processor_throws_exception() -> None:
    # create an iterator over some number of rows
    def get_row_iter() -> Iterator[DatasetRowDescription]:
        for i in range(100):
            yield (i, {"my_id": str(uuid.uuid4())})

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        dataset_writer_instance = DatasetWriter(
            "test",
            tmp_path,
            get_row_iter(),
            _row_processor_that_throws_exception,
            max_processes=2,
        )

        with pytest.raises(RuntimeError) as exc:
            dataset_df = dataset_writer_instance.create_dataset()
            assert str(exc.value) == "Row Processor Error For Test"

        # the row processor failed, so there should not be any residual files
        # that leaves the dataset in an inconsistent state
        assert len(list(tmp_path.rglob("*"))) == 0
