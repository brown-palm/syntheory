import warnings
from typing import List, Iterator, Iterable, Dict, Any
from pathlib import Path
from bisect import bisect
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import (
    PITCH_CLASS_TO_NOTE_NAME_SHARP,
    PITCH_CLASS_TO_NOTE_NAME_ENHARMONIC,
)
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_melody,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_REGISTERS = (3, 6, 9)


def get_note_midi(
    midi_note_val: int,
    num_plays: int = 4,
    play_duration_in_beats: int = 2,
):
    notes = []
    prev_beat = 0
    for _ in range(num_plays):
        notes.append(
            (prev_beat, prev_beat + play_duration_in_beats, (midi_note_val, None))
        )
        prev_beat += play_duration_in_beats
    return notes


def get_register(midi_note_value: int) -> int:
    # 3 registers: low, mid, high
    # 3 octaves each
    octave = midi_note_value // 12
    return bisect(_REGISTERS, octave)


def get_note_name_from_pitch_class(pitch_class: int) -> str:
    return PITCH_CLASS_TO_NOTE_NAME_SHARP[pitch_class]


def get_all_midi_note_values() -> Iterator[int]:
    # [0, 127] are MIDI note values, from C-1 to G9
    # but the soundfont we use does not have audio above C9, and the final
    # octave, even if it did would not have an even distribution of
    # pitch classes (does not have G# - B)
    # return MIDI notes from 0 to 107. C0 -> B8
    return iter(range(108))


def get_row_iterator(
    midi_note_values: Iterable[int], instrument_infos: Iterable[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for midi_note_val in midi_note_values:
        note_name = get_note_name_from_pitch_class(midi_note_val % 12)
        register = get_register(midi_note_val)
        for instrument_info in instrument_infos:
            yield (
                idx,
                {
                    "instrument_info": instrument_info,
                    "midi_note_val": midi_note_val,
                    "register": register,
                    "note_name": note_name,
                },
            )
            idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    midi_note_val = row_info["midi_note_val"]
    note_name = row_info["note_name"]
    register = row_info["register"]

    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{midi_note_val}_{register}_{note_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{midi_note_val}_{register}_{note_name}_{midi_program_num}_{cleaned_name}.wav"
    )
    note_midi = get_note_midi(midi_note_val)
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
    write_melody(note_midi, midi_track, channel=2)
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)

    octave = midi_note_val // 12
    root_note_pitch_class = midi_note_val % 12

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "root_note_pitch_class": root_note_pitch_class,
                "octave": octave,
                "root_note_is_accidental": note_name.endswith("#"),
                "register": register,
                "midi_note_val": midi_note_val,
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
    """This requires 14.02 GB.
    
    This has 9,936 total samples in the default configuration.
    
    NOTE:
        There are 88 samples that are silent.
            88 / 9936 ~ < 1%
    """
    # configure the dataset
    dataset_name = "notes"
    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_row_iterator(
            get_all_midi_note_values(),
            get_instruments(
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
