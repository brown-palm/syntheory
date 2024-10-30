import warnings
from pathlib import Path
from typing import Tuple, List, Iterator, Dict, Any, Iterable
from itertools import product
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import MODES, PITCH_CLASS_TO_NOTE_NAME_SHARP
from dataset.music.transforms import get_scale, get_tonic_midi_note_value
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_melody,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent

_PLAY_STYLE = {
    0: "UP",
    1: "DOWN",
}


def get_scale_midi(
    root_note_name: str,
    scale_mode: str,
    play_style: int,
    include_octave_above: bool = True,
):
    pitch_classes = get_scale(root_note_name, scale_mode)
    midi_tonic_val = get_tonic_midi_note_value(pitch_classes[0])
    offsets = MODES[scale_mode]
    if include_octave_above:
        offsets += (12,)

    notes = []
    for i in range(len(offsets)):
        notes.append(midi_tonic_val + offsets[i])

    if play_style == 1:
        # go down instead
        notes.reverse()

    # add timing for MIDI write
    timed_notes = []
    time_per_note = 1
    prev_beat = 0
    for n in notes:
        start_beat = prev_beat
        end_beat = start_beat + time_per_note
        timed_notes.append((start_beat, end_beat, (n, None)))
        prev_beat = end_beat

    return timed_notes


def get_all_scales(
    for_modes: Tuple[str] = (
        "ionian",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "aeolian",
        "locrian",
    ),
) -> List[Tuple[str, str]]:
    note_names = list(PITCH_CLASS_TO_NOTE_NAME_SHARP.values())
    return list(product(note_names, for_modes))


def get_row_iterator(
    scales: Iterable[Tuple[str, str]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note, mode in scales:
        for play_style, play_style_name in _PLAY_STYLE.items():
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "play_style": play_style,
                        "play_style_name": play_style_name,
                        "root_note": root_note,
                        "mode": mode,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    play_style = row_info["play_style"]
    play_style_name = row_info["play_style_name"]
    root_note = row_info["root_note"]
    mode = row_info["mode"]
    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{root_note}_{mode}_{play_style_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{root_note}_{mode}_{play_style_name}_{midi_program_num}_{cleaned_name}.wav"
    )

    scale_midi = get_scale_midi(root_note, mode, play_style)
    midi_file = create_midi_file()
    midi_track = create_midi_track(
        # this information doesn't change the sound
        bpm=120,
        time_signature=(4, 4),
        key_root=root_note,
        track_name=midi_program_name,
        program=midi_program_num,
        channel=2,
    )
    write_melody(scale_midi, midi_track, channel=2)
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)
    is_silent = is_wave_silent(synth_file_path)

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": root_note,
                "mode": mode,
                "play_style": play_style,
                "play_style_name": play_style_name,
                "midi_program_num": midi_program_num,
                "midi_program_name": midi_program_name,
                "midi_category": midi_category,
                "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                # e.g. TimGM6mb.sf2
                "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                "is_silent": is_silent,
            },
        )
    ]


if __name__ == "__main__":
    """Requires 21.82 GB of disk space.
    
    This has 15,456 samples.
    """
    # configure the dataset
    dataset_name = "scales"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            scales=get_all_scales(),
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

    # create the dataset
    dataset_df = dataset_writer.create_dataset()

    # warn of any silent samples
    num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]  # noqa
    if num_silent_samples > 0:
        warnings.warn(
            f"In the dataset, there were {num_silent_samples} silent samples.",
            UserWarning,
        )
