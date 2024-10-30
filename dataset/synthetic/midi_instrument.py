import csv
from typing import List, Dict, Any
from pathlib import Path

_ATONAL_INSTRUMENT_CATEGORIES = {"Synth Effects", "Sound Effects"}
# not exactly polyphonic, but these are instruments that have prominent overtones that
# can make the root note a bit ambiguous
_POLYPHONIC_PROGRAMS = {
    # celesta
    8,
    # glockenspiel
    9,
    # tubular bells
    14,
    # accordian
    21,
    # tango accordion
    23,
    # timpani
    47,
    # fifths lead
    86,
}
# these are programs that have very expressive voices that can make the root note ambiguous
_ARTICULATE_INSTRUMENT_PROGRAMS = {
    # harmonica
    22,
    # guitar harmonics
    31,
    # orchestral hit
    55,
    # shakuhachi
    77,
    # whistle
    78,
}


def get_instruments(
    ignore_atonal: bool = False,
    ignore_polyphonic: bool = False,
    ignore_highly_articulate: bool = False,
    take_only_first_category: bool = False,
) -> List[Dict[str, Any]]:
    """Get a dictionary of all MIDI programs.

    List can be found here: https://www.ccarh.org/courses/253/handout/gminstruments/
    and here: https://en.wikipedia.org/wiki/General_MIDI
    """
    all_instruments = []
    with open(Path(__file__).parent / "instruments.csv") as f:
        reader = csv.reader(f, delimiter=",")
        # skip the header
        next(reader)

        prev_category = ""
        for row in reader:
            sound_category, str_program_number, sound_name = row

            if prev_category == sound_category and take_only_first_category:
                # skip later instruments in the same category
                continue

            program_number = int(str_program_number)
            if (
                ignore_highly_articulate
                and program_number in _ARTICULATE_INSTRUMENT_PROGRAMS
            ):
                continue

            if ignore_atonal and sound_category in _ATONAL_INSTRUMENT_CATEGORIES:
                # skip this sound
                continue

            if ignore_polyphonic and program_number in _POLYPHONIC_PROGRAMS:
                # skip this sound
                continue

            all_instruments.append(
                {
                    "category": sound_category,
                    "program": program_number,
                    "name": sound_name,
                }
            )
            prev_category = sound_category

    return all_instruments
