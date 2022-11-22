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

from gridtk.tools import get_array_job_slice
from tqdm import tqdm

from zff import utils
from zff.arguments import arg_parser
from zff.zff import zff_vad


def segment():
    """Computes a speech/non-speech segmentation on given audio files."""

    # Get arguments
    args = arg_parser()

    # Audio extensions
    ext = args.audio_extensions

    # Read wave paths in a list
    if args.wavfolder:
        wavlist = utils.read_all_files(args.wavfolder, ext)
    elif args.wavfile:
        wavlist = [args.wavfile]

    # Slice file for SGE grid engine if available
    wavlist = wavlist[get_array_job_slice(len(wavlist))]

    # Iterate
    print(f"Segmenting {len(wavlist)} file(s). Going through:")
    for wav in tqdm(wavlist):

        fname_short = wav.split("/")[-1][:-4]

        # Read audio
        sr, s = utils.load_audio(wav, args.sameplerate)

        # Get segments
        boundary = zff_vad(s, sr, args.spectral_entropy, args.theta_scale)

        # Smoothing decision which merges close proximity segments
        # and eliminates small duration outlier ones.
        boundary = utils.smooth_decision(
            boundary,
            sr,
            args.smoothing_merge_factor,
            args.smoothing_outlier_factor,
        )

        # Convert from sample to time domain
        segments = utils.sample2time(s, sr, boundary)

        # Save
        utils.save_segments(args.output_path, fname_short, segments)
