import os
from contextlib import nullcontext
import tempfile
import multiprocessing
from functools import partial
from typing import Iterator, Dict, Any, Callable, Tuple, List
from pathlib import Path
import shutil
import pandas as pd
from util import use_770_permissions

DatasetRowDescription = Tuple[int, Dict[str, Any]]


class DatasetWriter:
    """A helper class that handles the creation of SynTheory datasets."""

    def __init__(
        self,
        dataset_name: str,
        save_to_parent_directory: Path,
        row_iterator: Iterator[DatasetRowDescription],
        row_processor: Callable[
            [Path, DatasetRowDescription], List[DatasetRowDescription]
        ],
        write_with_770_permissions: bool = True,
        max_processes: int = 4,
    ) -> None:
        """The constructor of SynTheory dataset writer.

        Args:
            dataset_name: The name of the dataset, e.g. 'tempos', 'notes', etc.
            save_to_parent_directory: The parent directory of the dataset after generation. For example,
                if we set this value to 'home' and the dataset_name to 'tempos', then the dataset is saved
                to a folder located at: '/home/tempos/' if everything runs successfully.
            row_iterator: An iterator over row data that will be passed to the row processor function.
            row_processor: A function that takes in the dataset context and the row information. This must
                return a tuple of the row ID and the data to be written to the dataset csv. This can write
                and edit files on disk.
            write_with_770_permissions: If true, all files written by this class will be given permissions
                770. This can be useful when working in a shared cluster environment.
            max_processes: The maximum number of processes to use when producing the dataset. The dataset
                is constructed using Python's multiprocess pool.
        """
        if not isinstance(save_to_parent_directory, Path):
            raise ValueError(
                f"save_to_parent_directory must be a Path object. "
                f"It was given as type: {type(save_to_parent_directory)}"
            )
        self.dataset_name = dataset_name
        self.parent_directory = save_to_parent_directory

        dataset_path = self.parent_directory / dataset_name

        self.dataset_path = dataset_path
        self.info_csv_filepath = dataset_path / "info.csv"
        self.write_with_770_permissions = write_with_770_permissions
        self.row_iterator = row_iterator
        self.row_processor = row_processor
        self._file_permission_ctx = (
            use_770_permissions if self.write_with_770_permissions else nullcontext
        )
        self.max_processes = max_processes

    def get_dataset_as_pandas_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.info_csv_filepath)

    def create_dataset(self) -> pd.DataFrame:
        """Construct the dataset as specified by the row_iterator and row_processor function.

        This is a very side-effect-y function. It will write many samples to a temporary directory
        and then move it to final resting directory if no exceptions are thrown.

        Returns: A pandas dataframe that contains all the generated samples.
        """
        if self.dataset_path.exists():
            raise RuntimeError(
                f"A dataset folder at this location already exists! Check: {self.dataset_path}"
            )

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            result_df = self._create_dataset_inner_loop(Path(tmp_dir))

            if not self.dataset_path.exists():
                self.dataset_path.mkdir(parents=True)

            # if no exceptions thrown after creation, promote the tmp folder to the 'real' location
            with self._file_permission_ctx():
                new_location = (
                    self.dataset_path.absolute().parent / Path(tmp_dir).parts[-1]
                )
                shutil.move(tmp_dir, self.dataset_path.absolute().parent)
                os.rename(
                    str(new_location.absolute()),
                    str((new_location.parent / self.dataset_name).absolute()),
                )
                tmp_dir = None

                # return the dataset dataframe
                return result_df

        except Exception as e:
            # the row processor or something it called raise an exception
            raise e
        finally:
            # if some exception was caught, then we should delete the temporary directory
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    def _create_dataset_inner_loop(self, tmp_output_path: Path) -> pd.DataFrame:
        # lambda functions cannot be pickled
        row_processor_func = partial(self.row_processor, tmp_output_path)

        with self._file_permission_ctx():
            rows = []
            with multiprocessing.Pool(self.max_processes) as pool:
                # gather the rows
                for row_set in pool.imap(row_processor_func, self.row_iterator):
                    # we could also just have the row ID be a property of the object returned, but
                    # we do it this way to draw attention to the ordering from the implementer
                    for row in row_set:
                        row_id, row_obj = row
                        rows.append({"row_id": row_id, **row_obj})

            # parse as a pandas array
            df = pd.json_normalize(rows)

            # could also make this an index but this is a bit clearer
            df = df.sort_values(by=["row_id"])

            # save to disk
            df.to_csv(tmp_output_path / self.info_csv_filepath.parts[-1], index=False)

            return df
