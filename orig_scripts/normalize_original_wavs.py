#!/usr/bin/env python3
from pathlib import Path

import os
import subprocess
import soundfile as sf
import pyloudnorm as pyln
import sys
import glob

# TODO: Wrap with function so we can import it
# For now, we can simply execute it within the Colab notebook.

rnn = "/content/rnnnoise/examples/rnnnoise_demo"
orig_wavs = "/content/drive/MyDrive/original"

paths = glob.glob(os.path.join(orig_wavs, "*.wav"))

for filepath in paths:
    target_filepath = Path(str(filepath).replace("original", "converted"))
    target_dir = os.path.dirname(target_filepath)
    # print(target_dir)
    if str(filepath) == str(target_filepath):
        raise ValueError("Source and target path are identical: {}".format(filepath))
    print("From: {}".format(filepath))
    print("To: {}".format(target_filepath))

    # Stereo to Mono; upsample to 48000 Hz
    # added -G to fix gain, -v 0.95

    subprocess.run(["sox", "-G", "-v", "0.95", filepath, "48k.wav", "remix", "-", "rate", "48000"])
    subprocess.run(["sox", "48k.wav", "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", "-t", "raw", "temp.raw"]) # convert wav to raw
    subprocess.run([rnn, "temp.raw", "rnn.raw"]) # apply rnnoise
    subprocess.run(["sox", "-G", "-v", "0.95", "-r", "48k", "-b", "16", "-e", "signed-integer", "rnn.raw", "-t", "wav", "rnn.wav"]) # convert raw back to wav

    subprocess.run(["mkdir", "-p", str(target_dir)])
    subprocess.run(["sox", "rnn.wav", str(target_filepath), "remix", "-", "highpass", "100", "lowpass", "7000", "rate", "22050"]) # apply high/low pass filter and change sr to 22050Hz
    data, rate = sf.read(target_filepath)

    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(data, -1.0)

    # measure the loudness first
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -25 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -25.0)
    sf.write(target_filepath, data=loudness_normalized_audio, samplerate=22050)
    print("")
