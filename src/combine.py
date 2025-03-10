import torch
from torchaudio.transforms import Resample
import torch.nn.functional as F

import comfy.model_management

from typing import Tuple
from ._types import AUDIO


class AudioCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "method": (
                    ["add", "mean", "subtract", "multiply", "divide"],
                    {
                        "default": "add",
                        "tooltip": "The method used to combine the audio waveforms.",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Combine two audio tracks by overlaying their waveforms."

    def is_valid_audio(self, audio: AUDIO) -> bool:
        return (
                audio is not None and
                "waveform" in audio and
                audio["waveform"].numel() > 0 and
                not torch.all(audio["waveform"] == 0)
        )

    def create_safe_silence(self, length: int, sample_rate: int) -> AUDIO:
        noise = torch.randn(1, length) * 1e-6  # Tiny noise floor
        return {"waveform": noise, "sample_rate": sample_rate}

    def main(
            self,
            audio_1: AUDIO = None,
            audio_2: AUDIO = None,
            method: str = "add",
    ) -> Tuple[AUDIO]:

        device = torch.device("cpu")

        valid_audio_1 = self.is_valid_audio(audio_1)
        valid_audio_2 = self.is_valid_audio(audio_2)

        if not valid_audio_1 and not valid_audio_2:
            length = 16000 * 5  # 5 seconds of safe silence
            return (self.create_safe_silence(length, 16000),)

        if valid_audio_1 and not valid_audio_2:
            waveform = audio_1["waveform"].to(device)
            if torch.all(waveform == 0):
                waveform += torch.randn_like(waveform) * 1e-6
            sample_rate = audio_1["sample_rate"]
            return ({"waveform": waveform, "sample_rate": sample_rate},)

        if valid_audio_2 and not valid_audio_1:
            waveform = audio_2["waveform"].to(device)
            if torch.all(waveform == 0):
                waveform += torch.randn_like(waveform) * 1e-6
            sample_rate = audio_2["sample_rate"]
            return ({"waveform": waveform, "sample_rate": sample_rate},)

        waveform_1 = audio_1["waveform"].to(device)
        input_sample_rate_1 = audio_1["sample_rate"]

        waveform_2 = audio_2["waveform"].to(device)
        input_sample_rate_2 = audio_2["sample_rate"]

        if input_sample_rate_1 != input_sample_rate_2:
            if input_sample_rate_1 < input_sample_rate_2:
                resample = Resample(input_sample_rate_1, input_sample_rate_2).to(device)
                waveform_1 = resample(waveform_1)
                output_sample_rate = input_sample_rate_2
            else:
                resample = Resample(input_sample_rate_2, input_sample_rate_1).to(device)
                waveform_2 = resample(waveform_2)
                output_sample_rate = input_sample_rate_1
        else:
            output_sample_rate = input_sample_rate_1

        min_length = min(waveform_1.shape[-1], waveform_2.shape[-1])
        waveform_1 = waveform_1[..., :min_length]
        waveform_2 = waveform_2[..., :min_length]

        match method:
            case "add":
                waveform = (waveform_1 + waveform_2)
            case "subtract":
                waveform = (waveform_1 - waveform_2)
            case "multiply":
                waveform = (waveform_1 * waveform_2)
            case "divide":
                waveform = (waveform_1 / waveform_2)
            case "mean":
                waveform = ((waveform_1 + waveform_2) / 2)

        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            {
                "waveform": waveform,
                "sample_rate": output_sample_rate,
            },
        )
