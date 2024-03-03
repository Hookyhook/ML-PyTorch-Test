import os.path

import matplotlib.pyplot as plt
import torch
import torchaudio



_SAMPLE_DIR = "_assets"

YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")

os.makedirs(YESNO_DATASET_PATH, exist_ok=True)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform.numpy()
    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()

dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

i = 1

waveform, sample_rate, label = dataset[i]


plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
