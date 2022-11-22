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


import argparse
import errno
import os


def arg_parser():
    parser = argparse.ArgumentParser(
        description="""
        Segment one or multiple audio files into `.csv` files,
        which list all the detected speech segments as [START, END].
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-w",
        "--wavfile",
        type=file_path,
        help="""
        Path to an audio file, e.g. `path/to/my/audiofile.wav`.
        """,
    )
    group.add_argument(
        "-f",
        "--wavfolder",
        type=dir_path,
        help="""
        Path to a folder containing multiple wav files,
        e.g. `path/to/folder/of/wav/files`.
        """,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        nargs="?",
        const=os.path.join(os.getcwd(), "segments"),
        default=os.path.join(os.getcwd(), "segments"),
        type=output_dir_path,
        help="""
        Path of the folder to save predictions.
        By default it will be make a `segments` folder in
        the root directory and save the segments there.
        These will contain .csv files demarking
        the [START, END] of the detected speech segments.
        """,
    )
    parser.add_argument(
        "-sr",
        "--sameplerate",
        default="None",
        type=str,
        help="""
        The desired sampling frequency of the audio files.
        By default `None` is used as value to keep native sampling rate.
        """,
    )
    parser.add_argument(
        "-e",
        "--audio-extensions",
        nargs="+",
        default=["wav"],
        help="""
        The type of audio files to look for a given directory,
        e.g. 'aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'.
        Default is just 'wav'.
        """,
    )
    parser.add_argument(
        "-smf",
        "--smoothing-merge-factor",
        default=5,
        type=int,
        help="""
        The window size ratio factor used for smoothing the final
        decision boundary by merging close proximity segments.
        Default is `5`, giving window size of `sr/5`.
        """,
    )
    parser.add_argument(
        "-sof",
        "--smoothing-outlier-factor",
        default=10,
        type=int,
        help="""
        The window size ratio factor used for smoothing the final decision
        boundary by removing small duration outlier segments.
        Default is `10`, giving a window size of `sr/10`.
        """,
    )
    parser.add_argument(
        "-t",
        "--theta-scale",
        default=1.0,
        type=float,
        help="""
        The factor used to scale the final threshold `theta`.
        Useful to strongly differentiate final decision surface with the threshold.
        Default is `1.0`.
        """,
    )
    parser.add_argument(
        "-ent",
        "--spectral-entropy",
        default=False,
        type=bool,
        help="""
        Weighs the ZFF evidences with the spectral entropy of the given audio
        signal to further highlight speech regions.
        Useful in noisy conditions. Default is `False`.
        """,
    )
    args = parser.parse_args()
    return args


def dir_path(path):
    if os.path.isdir(path):
        return path
    elif path is None:
        print(
            """
            Please provide an input directory containing
            audio files or else a single audio file.
            """
        )
    else:
        print(f"The directory {path} does not exist!")
        raise NotADirectoryError(path)


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        print(f"The file {path} does not exist!")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def output_dir_path(path):
    if path == os.path.join(os.getcwd(), "segments"):
        if not os.path.isdir(path):
            os.makedirs(path)
            return path
        else:
            return path
    else:
        if os.path.isdir(path):
            return path
        else:
            print(f"The directory {path} does not exist!")
            raise NotADirectoryError(path)
