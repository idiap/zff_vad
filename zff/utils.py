# Copyright (c) 2022, Idiap Research Institute. All rights reserved.
#
# This work is made available under GPLv3 license.
#
# Written by:
# Eklavya Sarkar <eklavya.sarkar@idiap.ch>
#
# Code based on the paper:
# "Unsupervised Voice Activity Detection by Modeling Source
# and System Information using Zero Frequency Filtering"
# Authors: Eklavya Sarkar, RaviShankar Prasad, Mathew Magimai Doss

import os

from pathlib import Path

import librosa
import numpy as np

from zff.zff import buffer


def load_audio(fileName, sr=None, dtype="float64"):
    """Loads an audio file with librosa.

    Parameters
    ----------
    fileName_ : str
        Path to an audio file.
    sr : int
        Sampling frequency of the audio file given with `fileName`.

    Returns
    -------
    tuple
        Returns a tuple of (sampling frequency, audio signal)
    """
    if sr == "None":
        sr = None
    elif sr is None:
        pass
    else:
        sr = int(sr)
    sig, fs = librosa.load(fileName, mono=True, sr=sr, dtype=dtype)
    sig /= np.max(np.abs(sig))
    return fs, sig


def read_all_files(wavfolder, ext):
    """Reads all `.wav` files given in `wavfolder`.

    Parameters
    ----------
    wavfolder : str
        Path to the folder containing `.wav` files.

    Returns
    -------
    list
        Returns a list of paths to all the `.wav` files.
    """
    return librosa.util.find_files(wavfolder, ext=ext)


def save_segments(output_path, fname_short, segments):
    """Saves the predicted segments as .csv file.

    Parameters
    ----------
    output_path : str
        Directory path to save the generated .csv file.
    fname_short : str
        File name of the .csv which will be the same as the read audio file.
    segments : list
        List containing the predicted segments in seconds.
    """
    np.savetxt(
        os.path.join(output_path, fname_short + ".csv"),
        segments,
        delimiter=",",
        fmt="%s",
    )


def file_parent_dir():
    """Get the parent directory of this file"""
    return str(Path(__file__).resolve().parent)


def sample2time(s, fs, boundary):
    """Converts given `boundary` from sample domain
    to time domain.

    Parameters
    ----------
    s : numpy.ndarray
        Input signal.
    fs : int
        Sampling frequency of `s`.
    boundary : numpy.ndarray
        Calculated decision boundary.

    Returns
    -------
    numpy.ndarray
        List of segments in seconds format easily exportable as a .csv file.
    """
    indices = np.where(np.diff(boundary, prepend=np.nan))[0]
    fgt = boundary.shape[0] * (fs / s.shape[0])  # scale

    lst = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]

        start_s = start / fgt
        end_s = end / fgt

        segment = boundary[start:end]
        status = set(segment)  # 0 or 1

        if next(iter(status)) == 1:
            lst.append([start_s, end_s])

    return np.array(lst)


def smooth_decision(s, sr, mergeFactor=5, outlierFactor=10):
    """Smoothens the VAD decision signal.

    Parameters
    ----------
    x : numpy.ndarray
        The binary VAD signal.
    winA : int
        Window size using for smoothing `x` by merging close proximity segments.
    winB : int
        Window size using for smoothing `x` by removing small duration outlier segments.

    Returns
    -------
    numpy.ndarray
        Smoothed binary VAD signal.
    """
    winA = sr // mergeFactor
    winB = sr // outlierFactor
    x = np.copy(s)
    # Window A
    x[0] = 0
    x[-1] = 0
    aDiffLine = np.where(np.diff(x))[0]
    aDiffBuff = np.array(buffer(aDiffLine[1:-1], 2, 0, "nodelay"))
    idx = np.where((np.diff(aDiffBuff, axis=0) < winA) * 1)[1]
    aDiffBuff = aDiffBuff[:, idx]
    n = aDiffBuff.shape[1]
    for i in range(n):
        x[int(aDiffBuff[0, i]) - 1 : int(aDiffBuff[1, i]) + 1] = 1

    # Window B
    x[0] = 0
    x[-1] = 0
    aDiffLine = np.where(np.diff(x))[0]
    aDiffBuff = np.array(buffer(aDiffLine, 2, 0, "nodelay"))
    idx = np.where(abs(np.diff(aDiffBuff, axis=0) < winB) * 1)[1]
    for i in range(len(idx)):
        if (aDiffBuff[1, idx[i]] != 0) and (aDiffBuff[0, idx[i]] != 0):
            x[int(aDiffBuff[0, idx[i]]) - 1 : int(aDiffBuff[1, idx[i]]) + 1] = 0
    # Return
    return x
