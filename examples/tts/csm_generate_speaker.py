import argparse
from pathlib import Path
from transformers import MimiModel, AutoFeatureExtractor
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput

from scipy.io.wavfile import read
from scipy.signal import resample
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speaker reference file, used by llama-tts-csm example",)
    parser.add_argument(
        "--model-path", type=Path,
        help="custom Mimi model path (safetensors model). If not specified, will use the default model from Hugging Face hub",
    )
    parser.add_argument(
        "infile", type=Path,
        help="the wav input file to use for generating the speaker reference file",
        nargs="?",
    )
    # parser.add_argument(
    #     "outfile", type=Path,
    #     help="the output file, defaults to the input file with .codes suffix",
    #     nargs="?",
    # )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.infile is None:
        raise ValueError("Input file is required")

    if not args.infile.exists():
        raise FileNotFoundError(f"Input file {args.infile} not found")

    # if args.outfile is None:
    #     args.outfile = args.infile.with_suffix(".codes")

    model = MimiModel.from_pretrained(args.model_path or "kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path or "kyutai/mimi")

    inp_audio = read(args.infile)
    original_sample_rate = inp_audio[0]
    audio_data = inp_audio[1]

    # If stereo, get only the first channel
    if len(audio_data.shape) > 1 and audio_data.shape[1] >= 2:
        audio_data = audio_data[:, 0]

    # resample
    target_sample_rate = 24000
    number_of_samples = round(len(audio_data) * float(target_sample_rate) / original_sample_rate)
    resampled_audio = resample(audio_data, number_of_samples)
    resampled_audio = resampled_audio / max(np.max(np.abs(resampled_audio)), 1e-10)

    # pre-process the inputs
    audio_sample = np.array(resampled_audio, dtype=float)
    inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    print('inputs', inputs["input_values"], inputs["input_values"].shape)

    # encode
    encoder_outputs = model.encode(inputs["input_values"])
    assert isinstance(encoder_outputs, MimiEncoderOutput), "encoder_outputs should be of type MimiEncoderOutput"

    # output
    flattened_audio_codes = encoder_outputs.audio_codes.transpose(-1, -2).flatten()
    for i in range(0, len(flattened_audio_codes), 16):
        for code in flattened_audio_codes[i:i+16].tolist():
            print(f"{code:<5}", end=",")
        print()


if __name__ == '__main__':
    main()
