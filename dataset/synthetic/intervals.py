import warnings
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any
from itertools import product
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import PITCH_CLASS_TO_NOTE_NAME_SHARP
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_melody,
    write_progression,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_PLAY_STYLE = {
    0: "UP",
    1: "DOWN",
    2: "UNISON",
}


def write_interval_midi(
    base_midi_note_val: int,
    interval_size: int,
    play_style: int,
    midi_track,
    channel: int,
):
    # all intervals are computed relative to the base note midi value, going UP
    midi_note_vals = (base_midi_note_val, base_midi_note_val + interval_size)

    notes = []
    if play_style == 0:
        # UP
        prev_beat = 0
        for _ in range(4):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals[0], None)))
            notes.append((prev_beat + 1, prev_beat + 2, (midi_note_vals[1], None)))
            prev_beat += 2
        return write_melody(notes, midi_track, channel=channel)
    elif play_style == 1:
        # DOWN
        prev_beat = 0
        for _ in range(4):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals[1], None)))
            notes.append((prev_beat + 1, prev_beat + 2, (midi_note_vals[0], None)))
            prev_beat += 2
        return write_melody(notes, midi_track, channel=channel)
    elif play_style == 2:
        # UNISON
        prev_beat = 0
        for _ in range(4 * 2):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals, None, None)))
            prev_beat += 1

        return write_progression(notes, midi_track, channel=channel)


def get_note_name_from_pitch_class(pitch_class: int) -> str:
    return PITCH_CLASS_TO_NOTE_NAME_SHARP[pitch_class]


def get_base_note_midi_note_values() -> Iterator[int]:
    return iter(range(48 + 12, 59 + 1 + 12))


def get_interval_values() -> Iterator[int]:
    # return 1 to 12 (m2, to P8)
    return iter(range(1, 12 + 1))


def get_all_interval_midi_settings() -> List[Tuple[int, int]]:
    return list(product(get_base_note_midi_note_values(), get_interval_values()))


def get_row_iterator(
    intervals: List[Tuple[int, int]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for midi_base_note, midi_interval_val in intervals:
        note_name = get_note_name_from_pitch_class(midi_base_note % 12)
        for play_style in _PLAY_STYLE.keys():
            for instrument_info in instruments:
                play_style_name = _PLAY_STYLE[play_style]
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "play_style_name": play_style_name,
                        "play_style": play_style,
                        "note_name": note_name,
                        "midi_interval_val": midi_interval_val,
                        "midi_base_note": midi_base_note,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row
    note_name = row_info["note_name"]
    play_style = row_info["play_style"]
    play_style_name = row_info["play_style_name"]
    midi_interval_val = row_info["midi_interval_val"]
    midi_base_note = row_info["midi_base_note"]

    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]
    cleaned_name = midi_program_name.replace(" ", "_")

    midi_file_path = (
        dataset_path
        / f"{note_name}_{midi_interval_val}_{play_style_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{note_name}_{midi_interval_val}_{play_style_name}_{midi_program_num}_{cleaned_name}.wav"
    )

    midi_file = create_midi_file()
    midi_track = create_midi_track(
        # this information doesn't change the sound
        bpm=120,
        time_signature=(4, 4),
        key_root=note_name,
        track_name=midi_program_name,
        program=midi_program_num,
        channel=2,
    )
    write_interval_midi(
        midi_base_note,
        midi_interval_val,
        play_style,
        midi_track,
        channel=2,
    )
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "root_note_pitch_class": midi_base_note % 12,
                "interval": midi_interval_val,
                "play_style": play_style,
                "play_style_name": play_style_name,
                "midi_note_val": midi_base_note,
                "midi_program_num": midi_program_num,
                "midi_program_name": midi_program_name,
                "midi_category": midi_category,
                "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                # e.g. TimGM6mb.sf2
                "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                "is_silent": is_wave_silent(synth_file_path),
            },
        )
    ]


if __name__ == "__main__":
    """This takes ~56.1 GB of disk space.
    
    There are 39,744 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "intervals"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            intervals=get_all_interval_midi_settings(),
            instruments=get_instruments(
                ignore_atonal=True,
                ignore_polyphonic=True,
                ignore_highly_articulate=True,
                take_only_first_category=False,
            ),
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
            f"In the dataset, there were {num_silent_samples} silent samples.",
            UserWarning,
        )
