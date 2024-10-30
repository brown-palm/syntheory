from typing import Tuple, List, Any
from dataclasses import dataclass
from math import ceil

from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

from dataset.music.transforms import InvalidMusicDefinition


@dataclass
class ClickTrackConfig:
    channel: int
    midi_program_num: int
    downbeat_note_midi_value: int
    offbeat_note_midi_value: int
    name: str
    downbeat_note_midi_velocity: int = 100
    offbeat_note_midi_velocity: int = 75


REVERB_LEVELS = {
    0: 0,
    1: 63,
    2: 127,
}


SHEETSAGE_DRUM_CLICK = ClickTrackConfig(
    channel=9,
    midi_program_num=0,
    downbeat_note_midi_value=37,
    offbeat_note_midi_value=31,
    name="default",
    downbeat_note_midi_velocity=100,
    offbeat_note_midi_velocity=75,
)


def get_reverb_on_message(reverb_level: int, channel: int) -> Message:
    if reverb_level not in REVERB_LEVELS:
        raise ValueError(
            "Reverb Level must be 0, 1, 2. 0 Meaning no reverb, 2 meaning full reverb."
        )

    return Message(
        "control_change",
        control=91,
        value=REVERB_LEVELS[reverb_level],
        time=0,
        channel=channel,
    )


def create_midi_file(
    ticks_per_beat: int = 480,
) -> MidiFile:
    # create a MIDI file object
    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
    return midi_file


def create_midi_track(
    bpm: int,
    time_signature: Tuple[int, int],
    key_root: str,
    track_name: str = "Piano",
    program: int = 0,
    channel: int = 1,
) -> MidiTrack:
    # create a track and add it to the file
    track = MidiTrack()
    # within the track, add some metadata messages
    track.append(MetaMessage("track_name", name=track_name, time=0))
    track.append(
        MetaMessage(
            "time_signature",
            numerator=time_signature[0],
            denominator=time_signature[1],
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )

    # see:
    # https://github.com/mido/mido/blob/7bc7432d8e6024d800394ad7f1a66d8ffb879abe/mido/midifiles/meta.py#L37

    # in microseconds per quarter note, this is fixed to 4/4 on purpose
    midi_tempo = bpm2tempo(bpm, time_signature=(4, 4))
    track.append(MetaMessage("set_tempo", tempo=midi_tempo, time=0))

    # init the program
    track.append(Message("program_change", program=program, time=0, channel=channel))

    return track


def write_progression(
    progression: List[Any],
    midi_track,
    ticks_per_beat: int = 480,
    channel: int = 1,
) -> None:
    if not progression:
        return

    # write a 'rest' before the first notes start playing, otherwise the notes
    # play too early
    start_beat, end_beat, node_data = progression[0]

    t = max(int((start_beat - 1) * ticks_per_beat), 0)
    if t > 0:
        midi_track.append(
            Message("note_on", note=64, velocity=0, time=0, channel=channel)
        )
        midi_track.append(
            Message("note_off", note=64, velocity=0, time=t, channel=channel)
        )

    prev_chord = None

    # (start beat, end beat, MIDI notes, chord name, scale degree for root of chord)
    for chord in progression:
        start_beat, end_beat, note_data = chord
        midi_notes, _, _ = note_data

        if prev_chord and prev_chord[1] < start_beat:
            # write a rest
            x = int((start_beat - prev_chord[1]) * ticks_per_beat)
            midi_track.append(
                Message("note_on", note=64, velocity=0, time=0, channel=channel)
            )
            midi_track.append(
                Message("note_off", note=64, velocity=0, time=x, channel=channel)
            )

        # trigger ON notes
        for n in midi_notes:
            if not (0 <= n <= 127):
                raise InvalidMusicDefinition(f"Invalid MIDI note value. Got: {n}")

            midi_track.append(
                Message("note_on", note=n, velocity=64, time=0, channel=channel)
            )

        for i, n in enumerate(midi_notes):
            t = int((end_beat - start_beat) * ticks_per_beat) if i == 0 else 0
            midi_track.append(
                Message("note_off", note=n, velocity=64, time=t, channel=channel)
            )

        prev_chord = chord


def write_melody(
    melody: List[Any], midi_track, ticks_per_beat: int = 480, channel: int = 2
) -> None:
    if not melody:
        return

    # write a 'rest' before the first notes start playing, otherwise the notes
    # play too early
    start_beat, end_beat, note_data = melody[0]

    t = max(int((start_beat - 1) * ticks_per_beat), 0)
    if t > 0:
        midi_track.append(
            Message("note_on", note=64, velocity=0, time=0, channel=channel)
        )
        midi_track.append(
            Message("note_off", note=64, velocity=0, time=t, channel=channel)
        )

    prev_note = None

    for note in melody:
        start_beat, end_beat, note_data = note
        midi_note_val, _ = note_data

        t = int((end_beat - start_beat) * ticks_per_beat)

        if prev_note and prev_note[1] < start_beat:
            # write a rest
            x = int((start_beat - prev_note[1]) * ticks_per_beat)
            midi_track.append(
                Message("note_on", note=64, velocity=0, time=0, channel=channel)
            )
            midi_track.append(
                Message("note_off", note=64, velocity=0, time=x, channel=channel)
            )

        if not (0 <= midi_note_val <= 127):
            raise InvalidMusicDefinition(
                f"Invalid MIDI note value. Got: {midi_note_val}"
            )

        midi_track.append(
            Message("note_on", note=midi_note_val, velocity=64, time=0, channel=channel)
        )
        midi_track.append(
            Message(
                "note_off", note=midi_note_val, velocity=64, time=t, channel=channel
            )
        )

        prev_note = note


def is_compound_time_signature(time_signature: Tuple[int, int]) -> bool:
    beats_per_bar = time_signature[0]
    beat_division = time_signature[1]
    return beats_per_bar % 3 == 0 and beat_division != 4


def write_click(
    time_signature: Tuple[int, int],
    midi_track,
    total_beats: int,
    ticks_per_beat: int = 480,
    config: ClickTrackConfig = SHEETSAGE_DRUM_CLICK,
    reverb_level: int = 0,
) -> None:
    beats_per_bar = time_signature[0]
    beat_division = time_signature[1]
    is_compound = is_compound_time_signature(time_signature)
    time_unit = (
        ticks_per_beat // 2
        if is_compound
        else int(ticks_per_beat / (beat_division / 4))
    )

    total_beats = ceil(total_beats)

    # add reverb
    midi_track.append(get_reverb_on_message(reverb_level, config.channel))

    for b in range(total_beats + 1):
        downbeat = b % beats_per_bar == 0
        velocity = (
            config.downbeat_note_midi_velocity
            if downbeat
            else config.offbeat_note_midi_velocity
        )
        pitch = (
            config.downbeat_note_midi_value
            if downbeat
            else config.offbeat_note_midi_value
        )
        midi_track.append(
            Message(
                "note_on", note=pitch, velocity=velocity, time=0, channel=config.channel
            )
        )
        midi_track.append(
            Message(
                "note_off",
                note=pitch,
                velocity=velocity,
                time=time_unit,
                channel=config.channel,
            )
        )
