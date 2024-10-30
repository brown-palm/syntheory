from typing import List, Union, Optional, Tuple
from dataset.music.constants import (
    CHORD_TONALITY,
    CHORD_TYPE_TO_NAME,
    MIDDLE_C_MIDI_NOTE,
    MODES,
    NOTE_NAME_TO_PITCH_CLASS,
    NOTE_TYPE_TO_CHORD_NOTES_CONVERSION,
    PITCH_CLASS_TO_NOTE_NAME_SHARP,
)

MODIFIER_OPERATION = {
    "bb": -2,
    "b": -1,
    "#": 1,
    "##": 2,
}

PitchClass = int
MIDINote = int
Scale = Tuple[PitchClass, ...]
Mode = str
ScaleDegreeString = str  # e.g. 5, #5, b7, 1
ScaleDegree = int


class InvalidMusicDefinition(Exception):
    pass


def scale_degree_to_pitch_class(
    scale: Scale, mode: Mode, sd: ScaleDegreeString, octave: int
) -> Tuple[PitchClass, MIDINote, str]:
    midi_tonic_val = get_tonic_midi_note_value(scale[0])
    offsets = MODES[mode]

    if not sd:
        raise InvalidMusicDefinition("Scale Degree is blank.")

    scale_degree = int(sd[-1]) - 1
    pitch_class = scale[scale_degree]
    midi_note = midi_tonic_val + offsets[scale_degree]

    if len(sd) > 1:
        modifier = sd[0]
        op = MODIFIER_OPERATION[modifier]
        pitch_class += op
        midi_note += op

    midi_note += octave * 12

    return pitch_class, midi_note, sd


def get_scale(root_note: Union[str, ScaleDegree], mode: Mode) -> Scale:
    if not isinstance(mode, str):
        raise InvalidMusicDefinition(f"Mode must be a string! Got type: {type(mode)}")
    intervals = MODES[mode]
    if isinstance(root_note, str):
        root_pc = NOTE_NAME_TO_PITCH_CLASS[root_note]
    else:
        root_pc = root_note
    return tuple([(root_pc + i) % 12 for i in intervals])


def get_tonic_midi_note_value(tonic_pc: PitchClass) -> MIDINote:
    """Interpret this as middle C being equidistant to all other key centers."""
    if 0 <= tonic_pc <= 5:
        return MIDDLE_C_MIDI_NOTE + tonic_pc
    else:
        return MIDDLE_C_MIDI_NOTE - (12 - tonic_pc)


def get_chord(
    scale: Scale,
    mode: Mode,
    root: PitchClass,
    inversion: Optional[str],
    chord_type,
    extensions: List[ScaleDegree],
    borrowed: Optional[Mode],
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    offsets = MODES[mode]
    midi_tonic_val = get_tonic_midi_note_value(scale[0])

    if not isinstance(root, int):
        raise InvalidMusicDefinition(f"Root is not an int! Got: {root} ({type(root)})")

    if borrowed:
        scale = get_scale(scale[0], borrowed)
        offsets = MODES[borrowed]

    # get every other scale degree from the root up
    # go all the way up to the 11th
    note_pitch_classes = []
    note_midi_values = []
    for i in range(root - 1, root - 1 + 12, 2):
        scale_deg = i % 7

        # get the pitch classes
        note_pitch_classes.append(scale[scale_deg])

        # get the absolute node ID in MIDI (within only 2 octave range)
        # if smoother voice leading is on, then chord components past the 7th
        # will get placed below the 5th
        note_midi_values.append(midi_tonic_val + (offsets[scale_deg]))

    # get the notes required for the type of chord (triad, 7th, 9th, 11th, etc.)
    note_pcs = note_pitch_classes[
        slice(*NOTE_TYPE_TO_CHORD_NOTES_CONVERSION[chord_type])
    ]
    note_midi_values = note_midi_values[
        slice(*NOTE_TYPE_TO_CHORD_NOTES_CONVERSION[chord_type])
    ]

    # determine the chord tonality
    third = (12 + (note_pcs[1] - note_pcs[0])) % 12
    fifth = (12 + (note_pcs[2] - note_pcs[0])) % 12
    intervals = (third, fifth)

    if inversion == "6":
        note_pcs = note_pcs[1:] + [note_pcs[0]]
        note_midi_values = note_midi_values[1:] + [note_midi_values[0] + 12]
    elif inversion == "64":
        note_pcs = note_pcs[2:] + note_pcs[:2]
        note_midi_values = note_midi_values[2:] + [x + 12 for x in note_midi_values[:2]]

    ext_names = ""
    for extension in extensions:
        scale_deg = extension % 7
        note_pcs.append(scale[scale_deg])
        note_midi_values.append(midi_tonic_val + (offsets[scale_deg] + 12))
        ext_names = ext_names or "add"
        ext_names += str(extension)

    # create musician-friendly name for this chord
    root_note = PITCH_CLASS_TO_NOTE_NAME_SHARP[note_pcs[0]]
    tonality = CHORD_TONALITY[intervals]
    borrowed_from = f" ({borrowed})" if borrowed else ""
    chord_name = f"{root_note}{tonality}{CHORD_TYPE_TO_NAME[chord_type]}{inversion or ''}[{ext_names}{borrowed_from}]"

    # return it as a tuple of pitch classes, MIDI values, the musical name, and the root pitch class
    # and from which mode it is borrowed (if it is)
    return note_pcs, note_midi_values, chord_name, root, borrowed


def get_minor_triad(
    root_note: PitchClass, inversion: Optional[str]
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    # get aeolian of root
    scale = get_scale(root_note, "aeolian")
    return get_chord(
        scale,
        mode="aeolian",
        root=1,
        inversion=inversion,
        chord_type=5,
        extensions=[],
        borrowed=None,
    )


def get_major_triad(
    root_note: PitchClass, inversion: Optional[str]
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    # get ionian of root
    scale = get_scale(root_note, "ionian")
    return get_chord(
        scale,
        mode="ionian",
        root=1,
        inversion=inversion,
        chord_type=5,
        extensions=[],
        borrowed=None,
    )


def get_dom_7(
    root_note: PitchClass,
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    # get ionian of root
    scale = get_scale(root_note, "mixolydian")
    return get_chord(
        scale,
        mode="mixolydian",
        root=1,
        inversion=None,
        chord_type=7,
        extensions=[],
        borrowed=None,
    )


def get_diminished_triad(
    root_note: PitchClass, inversion: Optional[str]
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    # get locrian of root
    scale = get_scale(root_note, "locrian")
    return get_chord(
        scale,
        mode="locrian",
        root=1,
        inversion=inversion,
        chord_type=5,
        extensions=[],
        borrowed=None,
    )


def get_augmented_triad(
    root_note: PitchClass, inversion: Optional[str]
) -> Tuple[List[PitchClass], List[MIDINote], str, ScaleDegree, Optional[Mode]]:
    # get triad built on the III of the harmonic minor scale a m3 below the
    # given root
    root_pc = (root_note - 3) % 12
    scale = get_scale(root_pc, "harmonicMinor")
    return get_chord(
        scale,
        mode="harmonicMinor",
        root=3,
        inversion=inversion,
        chord_type=5,
        extensions=[],
        borrowed=None,
    )


def voice_midi_chord(note_midi_vals: List[MIDINote]) -> List[MIDINote]:
    # for now just drop the bottom note down an octave
    return [note_midi_vals[0] - 12, *note_midi_vals]
