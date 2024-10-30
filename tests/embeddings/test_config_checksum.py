import tempfile
from pathlib import Path

import pytest

from embeddings.config_checksum import compute_checksum


def test_compute_checksum() -> None:
    checksum = compute_checksum(b"abc123")
    assert (
        checksum == "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090"
    )

    checksum = compute_checksum(
        {
            "model_type": "JUKEBOX",
            "model_size": "L",
            "model_layer": 38,
            "concept": "tempos",
        }
    )
    assert (
        checksum == "dfe502a2ea415da8dfe7d17ae641cb7f314283941d41747445e2a64507726372"
    )

    with pytest.raises(ValueError) as exc:
        compute_checksum(b"abc123", algorithm="x")
        assert str(exc.value) == "Uknown algorithm."


def test_compute_checksum_for_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        text_file = tmp_path / "test.txt"
        text_file.write_text("hello world")

        # give a file as the argument for the checksum
        checksum = compute_checksum(text_file)
        assert (
            checksum
            == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
