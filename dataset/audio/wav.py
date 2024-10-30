import tempfile
import shutil
from typing import Union, Tuple, Optional
from pathlib import Path
import numpy as np
import random
import ffmpeg
import librosa


def is_wave_silent(file_path: Union[str, Path]) -> bool:
    """Returns true if the wav file at the given path is completely silent.

    Args:
        file_path: The path where the file exists on disk.

    Returns: True if the wav file is silent, False otherwise.
    """
    audio_data, _ = get_wav_as_numpy(file_path)
    return np.all(audio_data == 0)


def get_wav_as_numpy(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Loads a wav file at its original sample rate and returns it as numpy with its sample rate.
    Args:
        file_path: The location of the file to load

    Returns: A numpy array of samples in the wav with multi-channels flattened.
        Sample rate is not returned.
    """
    fp = str(file_path.absolute()) if isinstance(file_path, Path) else str(file_path)
    # load audio with original sample rate
    audio, sample_rate = librosa.load(fp, sr=None)
    return audio.flatten(), sample_rate


def random_trim(
    source_wav_path: Path,
    save_offset_wav_to_path: Path,
    target_duration: float,
    overwrite_output: bool = False,
    seed: Optional[int] = None,
) -> float:
    """Given a filepath where a valid wav file exists and a location to which to save a variation, randomly
    offset the audio at the input, trim it to a specific length, and save it at the output location.

    Args:
        source_wav_path: The audio wav file to offset randomly
        save_offset_wav_to_path: The file path where we want to save the offset. Warning: this will
            overwrite what exists at that path.
        target_duration: The desired output length of the randomly offset sample in seconds. This should
            be less than the total length of the sample.
        overwrite_output: If this is true, any file that exists at the target location will be automatically
            overwritten by calling this. If false, this throws an error. By default, this is false.
        seed: (optional) random seed for making deterministic offsets.

    Returns: The new start time (relative to the original audio sample) of the resulting randomly offset
        sample. For example, if the input was 5 seconds long, and we want the target to be 4 seconds, then
        this function might return 0.5 to represent that the randomly offset sample that it saved starts
        0.5 seconds into the original (and ends 0.5 seconds before the original ended).
    """
    if overwrite_output is False:
        if save_offset_wav_to_path.exists() and save_offset_wav_to_path.is_file():
            raise RuntimeError(
                f"overwrite_output is False and there already exists a file at the desired "
                f"save location: {save_offset_wav_to_path}."
            )

    probe = ffmpeg.probe(source_wav_path)
    audio_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    )
    duration = float(audio_info["duration"])

    if target_duration >= duration:
        raise ValueError(
            f"The target duration must be less than the duration of the sample. The given "
            f"target duration was: {target_duration}, but the sample is only of length {duration}."
        )

    if seed:
        random.seed(seed)

    # randomly offset the time
    start_time = random.uniform(0, duration - target_duration)

    # Use ffmpeg to offset the audio to this random start time and trim to the desired length.
    g = (
        ffmpeg.input(source_wav_path, ss=start_time, t=target_duration)
        .output(str(save_offset_wav_to_path.absolute()))
        .overwrite_output()
        .global_args("-nostdin")
    )
    try:
        g.run(capture_stderr=True)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode()
        raise RuntimeError(f"ffmpeg raise an error: {error_message}") from e

    # return the time that is the new start time after random offset
    return start_time


def trim(
    source_wav_path: Path,
    save_offset_wav_to_path: Path,
    target_duration: float,
    overwrite_output: bool = False,
) -> float:
    """Given a filepath where a valid wav file exists and a location to which to save a trimmed version,
    cut the end of the input sample to trim it to a specific length.

    Args:
        source_wav_path: The audio wav file to trim
        save_offset_wav_to_path: The file path where we want to save the offset.
        target_duration: The desired output length of the trimmed sample in seconds.
        overwrite_output: If this is true, any file that exists at the target location will be automatically
            overwritten by calling this. If false, this throws an error. By default, this is false.

    Returns: the amount of time in seconds removed from the end of the trim.
    """
    if overwrite_output is False:
        if save_offset_wav_to_path.exists() and save_offset_wav_to_path.is_file():
            raise RuntimeError(
                f"overwrite_output is False and there already exists a file at the desired "
                f"save location: {save_offset_wav_to_path}."
            )

    probe = ffmpeg.probe(source_wav_path)
    audio_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    )
    duration = float(audio_info["duration"])

    if target_duration > duration:
        raise ValueError(
            f"The target duration must be less than the duration of the sample. The given "
            f"target duration was: {target_duration}, but the sample is only of length {duration}."
        )

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=source_wav_path.suffix
    ) as tmp_file:
        # ffmpeg cannot edit files in place, so to get this functionality we just create a temp
        # file and then replace the original file with it if we want
        temp_file_path = tmp_file.name

        # Use ffmpeg to offset the audio to this random start time and trim to the desired length.
        g = (
            ffmpeg.input(source_wav_path, ss=0, t=target_duration)
            .output(temp_file_path)
            .overwrite_output()
            .global_args("-nostdin")
        )
        try:
            g.run(capture_stderr=True)
        except ffmpeg.Error as e:
            error_message = e.stderr.decode()
            raise RuntimeError(f"ffmpeg raise an error: {error_message}") from e

    shutil.move(temp_file_path, save_offset_wav_to_path)

    # return the time that is the new start time after random offset
    return duration - target_duration
