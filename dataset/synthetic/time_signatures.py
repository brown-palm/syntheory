import warnings
from pathlib import Path
from typing import Tuple, List, Iterable, Iterator, Optional
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.midi import ClickTrackConfig
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent, random_trim, trim
from dataset.music.track import create_click_track_midi
from dataset.music.midi import is_compound_time_signature
from dataset.synthetic.metronome_configs import CLICK_CONFIGS
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription


def get_all_time_signatures() -> List[Tuple[int, int]]:
    return [(2, 2), (2, 4), (3, 4), (3, 8), (4, 4), (6, 8), (9, 8), (12, 8)]


def get_row_iterator(
    time_signatures: Iterable[Tuple[int, int]],
    click_configs: Iterable[ClickTrackConfig],
    num_reverb_levels: int,
    num_random_offsets: int,
    target_duration_per_sample_in_sec: float,
    bpm: int = 120,
    seed: Optional[int] = None,
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for time_signature in time_signatures:
        for reverb_level in range(num_reverb_levels):
            for config in click_configs:
                yield (
                    idx,
                    {
                        "bpm": bpm,
                        "time_signature": time_signature,
                        "click_config": config,
                        "reverb_level": reverb_level,
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
    time_signature = row_info["time_signature"]
    reverb_level = row_info["reverb_level"]
    bpm = row_info["bpm"]
    num_random_offsets = row_info["num_random_offsets"]
    target_duration_per_sample_in_sec = row_info["target_duration_per_sample_in_sec"]
    midi_program_num = config.midi_program_num

    # play approx 30 seconds of audio
    total_beats_to_play = int((time_signature[1] / 4) * (bpm // 2))
    midi_file = create_click_track_midi(
        bpm,
        # ~ approximately 20 seconds but not exactly, we over-shoot it
        # so that we have space to move around after the wav is created
        total_beats_to_play,
        midi_file=None,
        time_signature=time_signature,
        config=config,
        reverb_level=reverb_level,
    )

    time_signature_readable_name = str(time_signature)[1:-1].replace(", ", "_")
    midi_file_path = (
        dataset_path
        / f"{time_signature_readable_name}_{bpm}_bpm_{config.name}_reverb_level_{reverb_level}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{time_signature_readable_name}_{bpm}_bpm_{config.name}_reverb_level_{reverb_level}.wav"
    )
    midi_file.save(midi_file_path)

    # play the MIDI, realizing it to a waveform
    produce_synth_wav_from_midi(midi_file_path, synth_file_path, show_logs=True)

    # force each sample to be 30 seconds
    trim(synth_file_path, synth_file_path, target_duration=30.0, overwrite_output=True)

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

        # record this row in the csv
        rows.append(
            (
                row_idx + i,
                {
                    "time_signature": time_signature,
                    "time_signature_beats": time_signature[0],
                    "time_signature_subdivision": time_signature[1],
                    "is_compound": int(is_compound_time_signature(time_signature)),
                    "bpm": bpm,
                    "click_config_name": config.name,
                    "midi_program_num": midi_program_num,
                    "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                    "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                    "offset_file_path": str(offset_path.relative_to(dataset_path)),
                    "offset_time": str(offset_time),
                    # e.g. TimGM6mb.sf2
                    "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                    "reverb_level": reverb_level,
                    "is_silent": is_silent,
                },
            )
        )
    return rows


if __name__ == "__main__":
    """This requires 1.48 GB.
    Contains 1,200 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "time_signatures"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            get_all_time_signatures(),
            click_configs=CLICK_CONFIGS,
            num_reverb_levels=3,
            num_random_offsets=10,
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
