# C D E F G A B C
IONIAN_INTERVALS = (0, 2, 4, 5, 7, 9, 11)

LYDIAN_INTERVALS = (
    # 1
    IONIAN_INTERVALS[0],
    # 2
    IONIAN_INTERVALS[1],
    # 3
    IONIAN_INTERVALS[2],
    # #4
    IONIAN_INTERVALS[3] + 1,
    # 5
    IONIAN_INTERVALS[4],
    # 6
    IONIAN_INTERVALS[5],
    # 7
    IONIAN_INTERVALS[6],
)

MIXOLYDIAN_INTERVALS = (
    # 1
    IONIAN_INTERVALS[0],
    # 2
    IONIAN_INTERVALS[1],
    # 3
    IONIAN_INTERVALS[2],
    # 4
    IONIAN_INTERVALS[3],
    # 5
    IONIAN_INTERVALS[4],
    # 6
    IONIAN_INTERVALS[5],
    # b7
    IONIAN_INTERVALS[6] - 1,
)

# C D Eb F G Ab Bb C
AEOLIAN_INTERVALS = (
    # 1
    IONIAN_INTERVALS[0],
    # 2
    IONIAN_INTERVALS[1],
    # b3
    IONIAN_INTERVALS[2] - 1,
    # 4
    IONIAN_INTERVALS[3],
    # 5
    IONIAN_INTERVALS[4],
    # b6
    IONIAN_INTERVALS[5] - 1,
    # b7
    IONIAN_INTERVALS[6] - 1,
)

HARMONIC_MINOR_INTERVALS = (
    # 1
    AEOLIAN_INTERVALS[0],
    # 2
    AEOLIAN_INTERVALS[1],
    # b3
    AEOLIAN_INTERVALS[2],
    # 4
    AEOLIAN_INTERVALS[3],
    # 5
    AEOLIAN_INTERVALS[4],
    # b6
    AEOLIAN_INTERVALS[5],
    # 7
    AEOLIAN_INTERVALS[6] + 1,
)

PHRYGIAN_INTERVALS = (
    # 1
    AEOLIAN_INTERVALS[0],
    # b2
    AEOLIAN_INTERVALS[1] - 1,
    # b3
    AEOLIAN_INTERVALS[2],
    # 4
    AEOLIAN_INTERVALS[3],
    # 5
    AEOLIAN_INTERVALS[4],
    # b6
    AEOLIAN_INTERVALS[5],
    # b7
    AEOLIAN_INTERVALS[6],
)

PHRYGIAN_DOMINANT_INTERVALS = (
    # 1
    IONIAN_INTERVALS[0],
    # b2
    IONIAN_INTERVALS[1] - 1,
    # 3
    IONIAN_INTERVALS[2],
    # 4
    IONIAN_INTERVALS[3],
    # 5
    IONIAN_INTERVALS[4],
    # b6
    IONIAN_INTERVALS[5] - 1,
    # b7
    IONIAN_INTERVALS[6] - 1,
)

LOCRIAN_INTERVALS = (
    # 1
    AEOLIAN_INTERVALS[0],
    # b2
    AEOLIAN_INTERVALS[1] - 1,
    # b3
    AEOLIAN_INTERVALS[2],
    # 4
    AEOLIAN_INTERVALS[3],
    # b5
    AEOLIAN_INTERVALS[4] - 1,
    # b6
    AEOLIAN_INTERVALS[5],
    # b7
    AEOLIAN_INTERVALS[6],
)

DORIAN_INTERVALS = (
    # 1
    IONIAN_INTERVALS[0],
    # 2
    IONIAN_INTERVALS[1],
    # b3
    IONIAN_INTERVALS[2] - 1,
    # 4
    IONIAN_INTERVALS[3],
    # 5
    IONIAN_INTERVALS[4],
    # 6
    IONIAN_INTERVALS[5],
    # b7
    IONIAN_INTERVALS[6] - 1,
)

MODES = {
    "aeolian": AEOLIAN_INTERVALS,
    "dorian": DORIAN_INTERVALS,
    "harmonicMinor": HARMONIC_MINOR_INTERVALS,
    "ionian": IONIAN_INTERVALS,
    "locrian": LOCRIAN_INTERVALS,
    "lydian": LYDIAN_INTERVALS,
    "major": IONIAN_INTERVALS,
    "minor": AEOLIAN_INTERVALS,
    "mixolydian": MIXOLYDIAN_INTERVALS,
    "phrygian": PHRYGIAN_INTERVALS,
    "phrygianDominant": PHRYGIAN_DOMINANT_INTERVALS,
}

NOTE_NAME_TO_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    # There are some analyses that report E# as the tonic
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

PITCH_CLASS_TO_NOTE_NAME_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

PITCH_CLASS_TO_NOTE_NAME_ENHARMONIC = {
    0: ("C",),
    1: ("C#", "Db"),
    2: ("D",),
    3: ("D#", "Eb"),
    4: ("E",),
    5: ("F",),
    6: ("F#", "Gb"),
    7: ("G",),
    8: ("G#", "Ab"),
    9: ("A",),
    10: ("A#", "Bb"),
    11: ("B",),
}

# defines a slice of notes up to which we want to put in the chord
NOTE_TYPE_TO_CHORD_NOTES_CONVERSION = {
    # 1 3 5
    5: (0, 3),
    # 1 3 5 7
    7: (0, 4),
    # 1 3 5 7 9
    9: (0, 5),
    # 1 3 5 7 9 11
    11: (0, 6),
    # 1 3 5 7 9 11 13
    13: (0, 7),
}

CHORD_TONALITY = {
    # Major: M3, P5
    (4, 7): "Maj",
    # Minor: m3, P5
    (3, 7): "m",
    # Dimished: m3, d5
    (3, 6): "d",
    # Augmented: M3, A5
    (4, 8): "A",
}

CHORD_TYPE_TO_NAME = {
    # triad is implied
    5: "",
    7: "7",
    9: "9",
    11: "11",
    13: "13",
}

# if middle C is equidistant from all others,
# there are 6 keys above it, 6 below it
# F# G G# A A# B
# C
# C# D D# E F
MIDDLE_C_MIDI_NOTE = 60
