from dataset.music.midi import ClickTrackConfig, SHEETSAGE_DRUM_CLICK


WOODBLOCK_DARK = ClickTrackConfig(
    channel=2,
    midi_program_num=115,
    downbeat_note_midi_value=37,
    offbeat_note_midi_value=31,
    downbeat_note_midi_velocity=100,
    offbeat_note_midi_velocity=75,
    name="woodblock_dark",
)
WOODBLOCK_LIGHT = ClickTrackConfig(
    channel=2,
    midi_program_num=115,
    downbeat_note_midi_value=52,
    offbeat_note_midi_value=48,
    downbeat_note_midi_velocity=100,
    offbeat_note_midi_velocity=75,
    name="woodblock_light",
)
TAIKO = ClickTrackConfig(
    channel=2,
    midi_program_num=116,
    downbeat_note_midi_value=52,
    offbeat_note_midi_value=48,
    downbeat_note_midi_velocity=100,
    offbeat_note_midi_velocity=75,
    name="taiko",
)
SYNTH_DRUM = ClickTrackConfig(
    channel=2,
    midi_program_num=118,
    downbeat_note_midi_value=52,
    offbeat_note_midi_value=48,
    downbeat_note_midi_velocity=100,
    offbeat_note_midi_velocity=75,
    name="synth_drum",
)

CLICK_CONFIGS = (
    SHEETSAGE_DRUM_CLICK,
    WOODBLOCK_DARK,
    WOODBLOCK_LIGHT,
    TAIKO,
    SYNTH_DRUM,
)
