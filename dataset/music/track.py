from typing import Tuple, Optional
from mido import MidiFile
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_click,
    ClickTrackConfig,
    SHEETSAGE_DRUM_CLICK,
)


def create_click_track_midi(
    bpm: int,
    num_beats: int,
    midi_file: Optional[MidiFile],
    time_signature: Tuple[int, int] = (4, 4),
    config: ClickTrackConfig = SHEETSAGE_DRUM_CLICK,
    reverb_level: int = 0,
) -> MidiFile:
    _midi_file = midi_file or create_midi_file()
    drum_midi_track = create_midi_track(
        bpm,
        time_signature,
        key_root="",
        track_name="click",
        channel=config.channel,
        program=config.midi_program_num,
    )
    write_click(
        time_signature,
        drum_midi_track,
        total_beats=num_beats,
        ticks_per_beat=_midi_file.ticks_per_beat,
        config=config,
        reverb_level=reverb_level,
    )
    _midi_file.tracks.append(drum_midi_track)

    return _midi_file
