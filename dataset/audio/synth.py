from typing import Optional
import subprocess
from pathlib import Path
from config import DEFAULT_SYNTH_BINARY_LOCATION, DEFAULT_SOUNDFONT_LOCATION


def produce_synth_wav_from_midi(
    midi_filepath: Path, save_wav_to: Optional[Path] = None, show_logs: bool = True
):
    if not save_wav_to:
        save_wav_to = midi_filepath.with_name(midi_filepath.stem + ".wav")

    kw = {"capture_output": True}
    if not show_logs:
        kw = {
            "capture_output": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
    try:
        # run the synth
        result = subprocess.run(
            [
                str(Path(DEFAULT_SYNTH_BINARY_LOCATION).absolute()),
                str(Path(DEFAULT_SOUNDFONT_LOCATION).absolute()),
                str(midi_filepath),
                str(save_wav_to),
            ],
            check=True,
            text=True,
            **kw,
        )
        if show_logs:
            print(result.stdout)
    except FileNotFoundError as e:
        print(f"Could not find {e}. Has the synth binary been compiled?")
    except subprocess.CalledProcessError as e:  # pragma: no cover
        print(f"Error running: {e}. stderr: {e.output}")