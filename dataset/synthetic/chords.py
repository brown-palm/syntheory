import warnings
from pathlib import Path
from typing import Optional, List, Iterator, Iterable, Dict, Any, Tuple
from itertools import product

from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import (
    PITCH_CLASS_TO_NOTE_NAME_SHARP,
    PITCH_CLASS_TO_NOTE_NAME_ENHARMONIC,
)
from dataset.music.transforms import (
    get_major_triad,
    get_minor_triad,
    get_augmented_triad,
    get_diminished_triad,
)
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_progression,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_CHORD_MAP = {
    "major": get_major_triad,
    "minor": get_minor_triad,
    "aug": get_augmented_triad,
    "dim": get_diminished_triad,
}


def get_chord_midi(
    root_note_pitch_class: int,
    kind: str,
    inversion: Optional[str],
    num_plays: int = 4,
    play_duration_in_beats: int = 2,
):
    f = _CHORD_MAP[kind]
    pitch_classes, midi_notes, chord_name, _, _ = f(root_note_pitch_class, inversion)
    progression = []
    prev_beat = 0
    for _ in range(num_plays):
        progression.append(
            (prev_beat, prev_beat + play_duration_in_beats, (midi_notes, None, None))
        )
        prev_beat += play_duration_in_beats
    return progression


def get_note_name_from_pitch_class(pitch_class: int) -> str:
    return PITCH_CLASS_TO_NOTE_NAME_SHARP[pitch_class]


def get_all_chords() -> List[Tuple[int, str]]:
    note_names = list(PITCH_CLASS_TO_NOTE_NAME_SHARP.keys())
    chord_types = list(_CHORD_MAP.keys())
    return list(product(note_names, chord_types))


def get_row_iterator(
    chords: Iterable[Tuple[int, str]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note_pitch_class, chord_type in chords:
        note_name = get_note_name_from_pitch_class(root_note_pitch_class)
        for inversion in [None, "6", "64"]:
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "inversion": inversion,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                        "chord_type": chord_type,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    # get soundfont information
    note_name = row_info["note_name"]
    inversion = row_info["inversion"]
    root_note_pitch_class = row_info["root_note_pitch_class"]
    chord_type = row_info["chord_type"]
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{note_name}_{chord_type}_{inversion or '5'}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{note_name}_{chord_type}_{inversion or '5'}_{midi_program_num}_{cleaned_name}.wav"
    )

    chord_midi = get_chord_midi(root_note_pitch_class, chord_type, inversion)
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
    write_progression(chord_midi, midi_track, channel=2)
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)

    is_silent = is_wave_silent(synth_file_path)

    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "chord_type": chord_type,
                "inversion": inversion or "5",
                "root_note_is_accidental": note_name.endswith("#"),
                "root_note_pitch_class": root_note_pitch_class,
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
    """Requires 18.7 GB of disk space.
    
    There are 13,248 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "chords"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            chords=get_all_chords(),
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
