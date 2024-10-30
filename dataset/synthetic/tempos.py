from pathlib import Path
import warnings
from typing import Iterator, Tuple, Iterable, List, Optional

from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.midi import ClickTrackConfig
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import random_trim, is_wave_silent, trim
from dataset.music.track import create_click_track_midi
from dataset.synthetic.metronome_configs import CLICK_CONFIGS
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription


def get_all_tempos(slowest: int, fastest: int) -> Iterator[int]:
    """Get an iterator of all BPMs within a range (inclusive).

    Args:
        slowest: The slowest BPM to return in the iterator
        fastest: The fastest BPM to return in the iterator

    Returns: an iterator over the range [slowest, fastest] of integer BPMs.
    """
    return iter(range(slowest, fastest + 1))


def create_midi_and_synth(
    dataset_path: Path, bpm: int, config: ClickTrackConfig
) -> Tuple[Path, Path]:
    # get soundfont information
    midi_file = create_click_track_midi(
        bpm,
        # ~ approximately 20 seconds but not exactly, we over-shoot it
        # so that we have space to move around after the wav is created
        num_beats=bpm // 3,
        midi_file=None,
        time_signature=(4, 4),
        config=config,
    )

    midi_file_path = dataset_path / f"{bpm}_bpm_{config.name}.mid"
    midi_file.save(midi_file_path)

    synth_file_path = dataset_path / f"{bpm}_bpm_{config.name}.wav"

    # play the MIDI, realizing it to a waveform
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)

    # force each sample to be 20 seconds
    trim(synth_file_path, synth_file_path, target_duration=20.0, overwrite_output=True)

    return midi_file_path, synth_file_path


def get_row_iterator(
    slowest_bpm: int,
    fastest_bpm: int,
    click_configs: Iterable[ClickTrackConfig],
    num_random_offsets: int,
    target_duration_per_sample_in_sec: float,
    seed: Optional[int] = None,
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for bpm in get_all_tempos(slowest_bpm, fastest_bpm):
        for config in click_configs:
            yield (
                idx,
                {
                    "bpm": bpm,
                    "click_config": config,
                    "num_random_offsets": num_random_offsets,
                    "target_duration_per_sample_in_sec": target_duration_per_sample_in_sec,
                    "seed": seed,
                },
            )
            idx += num_random_offsets


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row
    config = row_info["click_config"]

    midi_file_path, synth_file_path = create_midi_and_synth(
        dataset_path, row_info["bpm"], config
    )

    num_random_offsets = row_info["num_random_offsets"]
    target_duration_per_sample_in_sec = row_info["target_duration_per_sample_in_sec"]
    rows = []
    for i in range(num_random_offsets):
        # produce random trim from this sample
        offset_path = synth_file_path.parent / (
            synth_file_path.stem + f"_offset_{i}.wav"
        )
        offset_time = random_trim(
            synth_file_path,
            offset_path,
            target_duration=target_duration_per_sample_in_sec,
            overwrite_output=False,
            seed=row_info["seed"],
        )
        is_silent = is_wave_silent(offset_path)
        rows.append(
            (
                row_idx + i,
                {
                    "bpm": row_info["bpm"],
                    "click_config_name": config.name,
                    "midi_program_num": config.midi_program_num,
                    "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                    "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                    "offset_file_path": str(offset_path.relative_to(dataset_path)),
                    "offset_time": str(offset_time),
                    "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                    "is_silent": is_silent,
                },
            )
        )
    return rows


if __name__ == "__main__":
    """This requires 5.68 GB of space.
    Contains 4,025 samples in the default configuration. 
    """
    # configure the dataset
    dataset_name = "tempos"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            slowest_bpm=50,
            fastest_bpm=210,
            click_configs=CLICK_CONFIGS,
            num_random_offsets=5,
            target_duration_per_sample_in_sec=4.0,
        ),
        row_processor=row_processor,
        max_processes=8,
    )

    # check the resulting info csv / dataframe
    dataset_df = dataset_writer.create_dataset()

    # warn of any silent samples
    num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]  # noqa
    if num_silent_samples > 0:
        warnings.warn(
            f"In the {dataset_name} dataset, there were {num_silent_samples} silent samples.",
            UserWarning,
        )
