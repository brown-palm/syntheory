from typing import Union, Dict, Any
from pathlib import Path
import hashlib
import gzip
import json


def compute_checksum(
    data_to_hash: Union[Path, bytes, Dict[str, Any]],
    algorithm: str = "sha256",
    gunzip: bool = False,
    chunk_size: int = 4096,
) -> str:
    """Computes checksum of target path.

    From SheetSage: https://github.com/chrisdonahue/sheetsage/blob/main/sheetsage/utils.py

    Parameters
    ----------
    data_to_hash : :class:`pathlib.Path` or bytes or dict
       Location, bytes of file, or dictionary to compute checksum for.
    algorithm : str, optional
       Hash algorithm (from :func:`hashlib.algorithms_available`); default ``sha256``.
    gunzip : bool, optional
       If true, decompress before computing checksum.
    chunk_size : int, optional
       Chunk size for iterating through file.

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown path.
    :class:`IsADirectoryError`
       Path is a directory.
    :class:`ValueError`
       Unknown algorithm.

    Returns
    -------
    str
       Hex representation of checksum.
    """
    if algorithm not in hashlib.algorithms_guaranteed or algorithm.startswith("shake"):
        raise ValueError("Unknown algorithm")
    computed = hashlib.new(algorithm)
    if isinstance(data_to_hash, bytes):
        computed.update(data_to_hash)
    elif isinstance(data_to_hash, dict):
        computed.update(
            json.dumps(data_to_hash, indent=2, sort_keys=True).encode("utf-8")
        )
    else:
        open_fn = gzip.open if gunzip else open
        with open_fn(data_to_hash, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                computed.update(data)
    return computed.hexdigest()
