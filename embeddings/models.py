from typing import Union, Optional
from enum import Enum
from pathlib import Path

import numpy as np

import librosa as lr
from librosa.feature import melspectrogram, chroma_cqt, mfcc

import jukemirlib
from transformers import MusicgenForConditionalGeneration, AutoProcessor

SAMPLE_RATE_FEATS = 22050 # librosa default sample rate for handcrafted features

DURATION_IN_SEC = 4.0


class Model(Enum):
    JUKEBOX = 1
    MUSICGEN_AUDIO_ENCODER = 2
    MUSICGEN_DECODER_LM_S = 3
    MUSICGEN_DECODER_LM_M = 4
    MUSICGEN_DECODER_LM_L = 5
    MELSPEC = 6
    CHROMA = 7
    MFCC = 8
    HANDCRAFT = 9

    def to_string(self) -> str:
        return self.name

    @property
    def max_layers(self) -> Optional[int]:
        if self == Model.JUKEBOX:
            return 72
        elif self == Model.MUSICGEN_DECODER_LM_S:
            return 24
        elif self in {Model.MUSICGEN_DECODER_LM_M, Model.MUSICGEN_DECODER_LM_L}:
            return 48
        elif self in {Model.MUSICGEN_AUDIO_ENCODER, Model.MELSPEC, Model.CHROMA, Model.MFCC, Model.HANDCRAFT}:
            return None
        else:
            raise ValueError(f"Invalid model: {self}")


def load_musicgen_model(model: Model):
    """
    Load MusicGen processor and model.
    """
    if model == Model.MUSICGEN_DECODER_LM_S:
        return AutoProcessor.from_pretrained(
            "facebook/musicgen-small"
        ), MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    elif model == Model.MUSICGEN_DECODER_LM_M:
        return AutoProcessor.from_pretrained(
            "facebook/musicgen-medium"
        ), MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
    elif (
        model == Model.MUSICGEN_AUDIO_ENCODER or model == Model.MUSICGEN_DECODER_LM_L
    ):
        return AutoProcessor.from_pretrained(
            "facebook/musicgen-large"
        ), MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
    else:
        raise ValueError(f"Not MusicGen model: {model}")


def load_audio(fpath: str, sr: int, duration: float) -> np.ndarray:
    audio, _ = lr.load(fpath, sr=sr, duration=duration)
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def concat_features(features):
    moments = []
    for i in range(3):
        f = np.diff(features, n=i, axis=1)
        moments.append(f.mean(axis=1))
        moments.append(f.std(axis=1))
    embedding = np.concatenate(moments)
    return embedding


def audio_file_to_embedding_np_array(
    audio_file: Path,
    model_type: Model = Model.JUKEBOX,
    processor: AutoProcessor = None,
    model: Union[MusicgenForConditionalGeneration] = None,
    extract_from_layer: Optional[int] = None,
    decoder_hidden_states: bool = True,
    meanpool: bool = True
) -> np.ndarray:
    # Jukebox Features
    if model_type == Model.JUKEBOX:
        if extract_from_layer is None:
            layers = list(range(1, 73))
        else: 
            layers = [extract_from_layer]

        reps = jukemirlib.extract(
            fpath=audio_file,
            layers=layers,
            duration=DURATION_IN_SEC,
            meanpool=True,
            # downsample to rate 15 using method "librosa_fft"
            downsample_target_rate=15,
            downsample_method=None,
        )
        # jukemirlib produces a dictionary where the key is the number of layers
        # but we want this as just a numpy array
        if extract_from_layer is None:
            embedding: np.ndarray = np.stack([reps[i] for i in layers])
        else:
            embedding: np.ndarray = reps[extract_from_layer]
        jukemirlib.lib.empty_cache()

    # Handcrafted features
    elif model_type in {Model.MELSPEC, Model.CHROMA, Model.MFCC, Model.HANDCRAFT}:
        audio = load_audio(audio_file, 22050, DURATION_IN_SEC)
        if model_type == Model.HANDCRAFT:
            embedding = np.concatenate([concat_features(melspectrogram(audio, sr=22050)),
                                        concat_features(chroma_cqt(audio, sr=22050)),
                                        concat_features(mfcc(audio, sr=22050))])
        else:
            if model_type == Model.MELSPEC:
                features = melspectrogram(audio, sr=22050)
            elif model_type == Model.CHROMA:
                features = chroma_cqt(audio, sr=22050)
            elif model_type == Model.MFCC:
                features = mfcc(y=audio, sr=22050)
            
            # concatentate mean and std across time of features & their 1st and 2nd order differences
            embedding = concat_features(features)
        
    # MusicGen Features
    elif model_type == Model.MUSICGEN_AUDIO_ENCODER:
        embedding: np.ndarray = extract_musicgen_audio_encoder_emb(
            audio_file, processor, model
        )

    elif model_type in {
        Model.MUSICGEN_DECODER_LM_S,
        Model.MUSICGEN_DECODER_LM_M,
        Model.MUSICGEN_DECODER_LM_L
    }:
        embedding: np.ndarray = extract_musicgen_decoder_lm_emb(
            audio_file,
            processor,
            model,
            extract_from_layer,
            hidden_states=decoder_hidden_states,
            meanpool=meanpool
        )

    else:
        raise ValueError(f"Invalid model: {model_type}")
    return embedding


def extract_musicgen_audio_encoder_emb(
    audio_file: Path, 
    processor: AutoProcessor, 
    model: Union[MusicgenForConditionalGeneration],
    meanpool: bool = True
) -> np.ndarray:
    """
    Extract embeddings from MusicGen Audio Encoder
    """
    # set up inputs
    sampling_rate = model.config.audio_encoder.sampling_rate  # MusicGen uses 32000 Hz

    audio = load_audio(str(audio_file), sampling_rate, DURATION_IN_SEC)

    inputs = processor(
        audio=audio,
        sampling_rate=sampling_rate,
        padding=True,
        return_tensors="pt",
    )

    x = inputs["input_values"]

    # audio encoder
    audio_encoder = model.get_audio_encoder()

    # extract representations from audio encoder
    for layer in audio_encoder.encoder.layers:
        x = layer(x)

    if meanpool:
        return x.mean(axis=2).squeeze().detach().numpy()
    else:
        return x.squeeze().detach().numpy()


def extract_musicgen_decoder_lm_emb(
    audio_file: Path,
    processor: AutoProcessor,
    model: Union[MusicgenForConditionalGeneration],
    extract_from_layer: Optional[int] = None,
    text_cond: str = "",
    hidden_states: bool = True,
    meanpool: bool = True
):
    """
    Extract embeddings from MusicGen Decoder LM
    """
    # set up inputs
    sampling_rate = model.config.audio_encoder.sampling_rate  # MusicGen uses 32000 Hz

    audio = load_audio(str(audio_file), sampling_rate, DURATION_IN_SEC)

    inputs = processor(
        audio=audio,
        text=text_cond,
        sampling_rate=sampling_rate,
        padding=True,
        return_tensors="pt",
    )

    # extract representations from decoder LM
    out = model(**inputs, output_attentions=True, output_hidden_states=True)

    # output decoder hidden states
    if hidden_states:
        if extract_from_layer is None:
            if meanpool:
                return np.stack(tuple(l.mean(axis=1).squeeze().detach().numpy() for l in out.decoder_hidden_states))
            else:
                return np.stack(tuple(l.squeeze().detach().numpy() for l in out.decoder_hidden_states))
        else:
            if meanpool:
                return out.decoder_hidden_states[extract_from_layer].mean(axis=1).squeeze().detach().numpy()
            else:
                return out.decoder_hidden_states[extract_from_layer].squeeze().detach().numpy()
    # output decoder attentions
    else:
        if extract_from_layer is None:
            if meanpool:
                return np.stack(tuple(l.mean(axis=(2, 3)).squeeze().detach().numpy() for l in out.decoder_attentions))
            else:
                return np.stack(tuple(l.squeeze().detach().numpy() for l in out.decoder_attentions))
        else:
            if meanpool:
                return out.decoder_attentions[extract_from_layer].mean(axis=(2, 3)).squeeze().detach().numpy()
            else:
                return out.decoder_attentions[extract_from_layer].squeeze().detach().numpy()
