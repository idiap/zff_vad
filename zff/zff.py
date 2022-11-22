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


import numpy as np

from scipy.fft import fft


def run_mean(sig, N, wintype="RECT"):
    """
    Computes and returns the moving average and sum of
    a window on the input signal `sig`.

    Parameters
    ----------
    sig : numpy.ndarray
        Input signal.
    N : float
        [description]
    wintype : str, optional
        Window type [Rectangular, Hamming, Hanning], by default 'RECT'

    Returns
    -------
    tuple
        The average and sum in the moving window.
    """

    M = len(sig)
    Nby2 = int(N / 2)
    Ntail = N - Nby2 - 1

    if wintype in ("REC", "RECT", "rec", "rect", "1"):
        h = np.ones(N)
    elif wintype in ("HAM", "HAMM", "ham", "hamm", "2"):
        h = np.hamming(N)
    elif wintype in ("HAN", "HANN", "han", "hann", "3"):
        h = np.hanning(N)
    else:
        print("Error : Unknown window type!")
        exit(1)

    x = np.convolve(sig, h)
    ysum = x[Nby2 : M + N - 1 - Ntail]

    xdiv = np.convolve(np.ones(len(sig)), h)
    x = x / xdiv
    yavg = x[Nby2 : M + N - 1 - Ntail]

    return yavg, ysum


def buffer(x, n, p=0, opt=None):
    """Buffer signal vector into an array of data frames.

    Source
    ------
    Based on: https://ch.mathworks.com/help/signal/ref/buffer.html
    Modified version of: https://stackoverflow.com/a/60020929

    Parameters
    ----------
    x: numpy.ndarray
        Signal array
    n: int
        Length of data segments (frames)
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) numpy.ndarray
        Buffer array created from X.
    """

    if opt not in ("nodelay", None):
        raise ValueError("{} not implemented".format(opt))

    i = 0
    if opt == "nodelay":
        # No zeros at array start
        result = x[:n]
        i = n
    else:
        # Start with `p` zeros
        result = np.hstack([np.zeros(p), x[: n - p]])
        i = n - p

    if len(x) < n:
        result = np.hstack([result, np.zeros(n - len(x))])

    # Make 2D array, cast to list for .append()
    result = list(np.expand_dims(result, axis=0))

    while i < len(x):
        # Create next column, add `p` results from last col if given
        col = x[i : i + (n - p)]
        if p != 0:
            col = np.hstack([result[-1][-p:], col])

        # Append zeros if last row and not length `n`
        if len(col):
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result.append(np.array(col))
        i += n - p

    return np.vstack(result).T


def gauss_win(N, a=2.5):
    """Exponentially scaled Gaussian window.

    Source
    -------
    Based on: https://ch.mathworks.com/help/signal/ref/gausswin.html

    Parameters
    ----------
    N : int
        Window length.
    a : float, optional
        Width factor, specified as a positive real scalar alpha.
        It is inversely proportional to the width of the window.

    Returns
    -------
    w : ndarray
        The scaled Gaussian window.
    """

    a = np.asarray(a)
    N = np.asarray(N)
    if np.any(N != np.round(N)) or np.any(N < 0):
        raise ValueError("Window length N must be a non-negative integer")
    if N.size == 0 or a.size == 0:
        return np.ones(N.shape)
    N = N.item()
    a = a.item()
    L = N - 1
    n = np.arange(0, L + 1) - L / 2
    w = np.exp(-(1 / 2) * (a * n / (L / 2)) ** 2)
    return w


def block_view(A, block_size):
    """
    Provides a 2D block view of a 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A.

    Parameters
    ----------
    A : numpy.ndarray
        Buffer.
    block_size : tuple
        Block size.

    Returns
    -------
    numpy.ndarray
        Strided view of the buffer `A`.
    """
    from numpy.lib.stride_tricks import as_strided as ast

    shape = (
        A.shape[0] // block_size[0],
        A.shape[1] // block_size[1],
    ) + block_size
    strides = (
        block_size[0] * A.strides[0],
        block_size[1] * A.strides[1],
    ) + A.strides
    return ast(A, shape=shape, strides=strides)


def autocorr(x):
    """Returns auto-correlation of a signal `x`.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.

    Returns
    -------
    numpy.ndarray
        Auto-correlation of input signal.
    """
    result = np.correlate(x, x, mode="full")
    result = result[result.size // 2 :]
    result = result / np.max(np.abs(result))
    return result


def xcorrWinLen(wav, fs):
    """Returns the cross-correlation of two discrete-time sequences.
    Cross-correlation measures the similarity between a vector x
    and shifted (lagged) copies of a vector y as a function of the lag.
    If x and y have different lengths, the function appends zeros to the
    end of the shorter vector so it has the same length as the other.

    Source
    -------
    Based on: https://ch.mathworks.com/help/matlab/ref/xcorr.html

    Parameters
    ----------
    wav : numpy.ndarray
        Input signal.
    fs : int
        Sampling frequency of the input signal `wav`.

    Returns
    -------
    _type_
        Cross-correlation of two discrete-time sequences.
    """
    zfSig = zeroFreqFilter(wav, fs, 2)
    zfSig = zfSig / np.max(np.abs(zfSig))

    frameSize = 30 * fs // 1000
    frameShift = 20 * fs // 1000

    en = np.convolve(zfSig**2, np.ones(frameSize))
    en = en[(frameSize // 2) - 1 : len(en) - frameSize // 2]
    en = en / frameSize
    en = np.sqrt(en)
    en = (en > max(en) / 5) * 1
    buff = buffer(zfSig, frameSize, frameShift, "nodelay")
    vad = np.sum(buffer(en, frameSize, frameShift, "nodelay"), axis=0)
    out = block_view(buff, (frameSize, 1)).squeeze()

    autocorr_out = []
    for row in out:
        autocorr_out.append(
            list(
                autocorr((row - np.mean(row)) * np.hamming(len(row)))
                / autocorr(np.hamming(len(row)))
            )
        )
    autocorr_out = np.array(autocorr_out).T

    minPitch = 3  # 2 ms == 500 Hz.
    maxPitch = 16  # 16 ms == 66.66 Hz.

    term = autocorr_out[
        (minPitch * fs // 1000) - 1 : (maxPitch * fs // 1000), :
    ]
    # maxv = term.max(0)
    maxi = np.argmax(term, 0)

    step = 0.5

    # Get bin centres, and convert them to edges for `np.histogram`
    x = np.arange(minPitch, maxPitch + 0.5, step) * fs // 1000 + 2
    x = list(x - np.diff(x)[0] // 2) + [x[-1] + np.diff(x)[0] // 2]

    # Get the y-values, and normalize
    pLoc = maxi[vad > frameSize * 0.8] + minPitch * fs // 1000
    y = np.histogram(pLoc, x)[0]
    y = y / len(pLoc)

    # val = y.max()
    idx = y.argmax()

    # Extra +1 because altho Matlab -> Python indices are different,
    # and we are using this number as an actual length
    idx = np.around(idx / 2) + minPitch + 1 + 1

    return idx


def remTrend(sig, winSize):
    """Remove trend from given signal `sig` using a window of
    size `winSize` and applying convolution.

    Parameters
    ----------
    sig : numpy.ndarray
        Input signal.
    winSize : [type]
        Window size used to compute the trend removal window.

    Returns
    -------
    numpy.ndarray
        The signal after processing it for trend-removal.
    """
    window = np.ones(winSize)
    rm = np.convolve(sig, window, "same")
    norm = np.convolve(np.ones(len(sig)), window, "same")
    rm = rm / norm
    out = sig - rm
    return out


def get_zff(x, fs, multiple_dx, zfsig_window_len=None):
    """Returns the ZFF and modZFF scores for the input signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal
    f : int
        Sampling frequency

    Returns
    -------
    tuple
        ZFF and modZFF scores.
    """
    if multiple_dx:
        dx = np.array(
            list(np.diff(np.diff(np.diff(np.diff(x))))) + [0, 0, 0, 0]
        )
    else:
        dx = np.array(list(np.diff(x)) + [0])

    z1, gci, _, _, winLen = zfsig(x, fs, zfsig_window_len)
    z2, _, _, _, _ = zfsig(x, fs, 0.2 * winLen)
    z3, _, _, _, _ = zfsig(x, fs, 0.1 * winLen)

    return z1, z2, z3, gci


def zfsig(wav, fs, winLength=None):
    """
    Returns a tuple of the ZFF signal `zf`, glottal closure instances `gci`,
    epoch strength `es`, fundamental frequency `f0`, and the window length `winLength`.

    Parameters
    ----------
    wav : numpy.ndarray
        The input signal.
    fs : int
        Sampling frequency of the input signal `wav`.
    winLength : int, optional
        Window length, by default `None`.

    Returns
    -------
    tuple
        ZFF signal, glottal closure instances, es,
        fundamental frequency, and the window length.
    """
    if winLength == None:
        winLength = xcorrWinLen(wav, fs)
        winLength = 1.2 * winLength
    zf = zeroFreqFilter(wav, fs, winLength)

    # Zero crossings
    zcs = np.where(np.diff((zf > 0)) == 1)[0]

    # Neg-to-pos zero crossings only
    gci = np.array([zc for zc in zcs if zf[zc - 1] < 0 and zf[zc + 1] > 0])
    es = np.abs(zf[gci + 1] - zf[gci - 1])
    T0 = np.diff(gci)
    T0 = T0 / fs
    f0 = 1.0 / T0

    return zf, gci, es, f0, winLength


def zeroFreqFilter(wav, fs, winLength):
    """Computes and returns the ZFF signal to the input `wav` signal.

    Parameters
    ----------
    wav : numpy.ndarray
        Input signal.
    fs : int
        Sampling frequency of `wav`.
    winLength : int
        Window length for trend removal.

    Returns
    -------
    _type_
        _description_
    """
    dwav = np.diff(wav)
    dwav = np.append(dwav, dwav[-1])
    dwav /= np.max(np.abs(dwav))
    N = len(dwav)

    zfSig = np.cumsum(np.cumsum(np.cumsum(np.cumsum(dwav))))
    winLength = round(winLength * fs / 1000)

    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)
    zfSig = remTrend(zfSig, winLength)

    zfSig[(N - winLength * 2) - 1 : N] = 0
    zfSig[0 : winLength * 2] = 0

    return zfSig


def a_zff_norm_sig(s1, fs, signal_length=10, zfsig_window_len=None):
    """Computes and returns the ZFF scores for VAD.

    Parameters
    ----------
    s1 : numpy.ndarray
        Raw input signal.
    f : int
        Sample rate of audio file.

    Returns
    -------
    numpy.ndarray
        ZFF scores on the input signal `s1` for VAD.
    """

    # Get ZFF values
    # If the signal longer than 10s, segment signal each 10s and
    # accumulate the ZFF and modified ZFF together
    if len(s1) / fs > signal_length:
        sB = buffer(s1, signal_length * fs, 0, "nodelay")
        _, n = sB.shape
        z1 = []
        z2 = []
        z3 = []
        for l1 in range(n):
            x = sB[:, l1]
            temp_z1, temp_z2, temp_z3, gci = get_zff(
                x, fs, multiple_dx=False, zfsig_window_len=zfsig_window_len
            )

            # Append
            z1.append(list(temp_z1))
            z2.append(list(temp_z2))
            z3.append(list(temp_z3))
    else:
        z1, z2, z3, gci = get_zff(
            s1, fs, multiple_dx=False, zfsig_window_len=zfsig_window_len
        )

    # Flatten
    z1 = np.array(z1).ravel()
    z2 = np.array(z2).ravel()
    z3 = np.array(z3).ravel()

    # Normalize
    z1 /= np.max(np.abs(z1))
    z2 /= np.max(np.abs(z2))
    z3 /= np.max(np.abs(z3))

    return z1, z2, z3, gci


def get_spectrum(s, fs, frameLen_s, frameShift_s):
    """Computes and returns the spectrum of an input signal `s`.

    Parameters
    ----------
    s : numpy.ndarray
        Input signal on which FFT is computed.
    fs : int
        Sampling frequency of the signal `s`.
    frameLen_s : int
        Frame length of `s` for buffering the signal.
    frameShift_s : int
        Frame shift of `s` for buffering the signal.

    Returns
    -------
    numpy.ndarray
        The spectrum of the input signal `s`.
    """
    frameLen = frameLen_s * fs // 1000
    frameShift = frameShift_s * fs // 1000
    b_x = buffer(s, frameLen, frameShift, "nodelay")
    f_x = fft(b_x, 1024, axis=0)
    x = np.abs(f_x[:512, :])
    return x


def specEnt(x):
    """Computes the entropy of the given spectrum signal `x`.

    Parameters
    ----------
    x : numpy.ndarray
        The spectrum of a given signal.

    Returns
    -------
    numpy.ndarray
        The entropy of input spectrum.
    """
    r, c = x.shape
    ent = np.zeros(c)
    for i in range(c):
        sp = x[:r, i] ** 2
        sp = sp / np.sum(sp, axis=-1, keepdims=True)
        min_nonzero = np.min(sp[np.nonzero(sp)])
        sp[sp == 0] = min_nonzero
        ent[i] = -np.sum(np.multiply(sp, np.log2(sp)), axis=-1)
    return ent


def getEntropy(s, fs, frameLen_s=30, frameShift_s=15):
    """Calculates the spectrum of the given signal `s`,
    and returns its entropy.

    Parameters
    ----------
    s : numpy.ndarray
        Input signal on which we compute its spectral entropy.
    fs : int
        Sampling frequency of the signal `s`.
    frameLen_s : int, optional
        Frame length of `s` for buffering the signal, by default 30
    frameShift_s : int, optional
        Frame shift of `s` for buffering the signal, by default 15

    Returns
    -------
    numpy.ndarray
        The spectral entropy of an input signal.
    """
    x = get_spectrum(s, fs, frameLen_s, frameShift_s)
    entropy = specEnt(x)
    return entropy


def create_th(x, winT=10 * 8):
    """Creates a threshold for the input decision surface `x`.

    Parameters
    ----------
    x : numpy.ndarray
        Decision surface on which a threshold is calculated.
    winT : int, optional
        Frame length for buffer, by default 10*8

    Returns
    -------
    numpy.ndarray
        Threshold calculated on the input decision surface.
    """
    bx = buffer(x, winT, 0, "nodelay")
    th = np.min(bx, axis=0) + (np.median(bx, axis=0) / 3)
    th = np.kron(np.ones((winT, 1)), th)
    th = th.T.flatten()
    return th[: len(x)]


def zff_cs(s, fs, verbose=False):
    """
    ZFF CS method returns the composite signal which is
    the sum of the y0, y1 and y2 signals, which capture
    f0 (source), F1 and F2 (system) information respectively.

    Parameters
    ----------
    s : numpy.ndarray
        Raw speech signal.
    fs : int
        Sampling frequency of `s`.
    verbose : bool

    Returns
    -------
    numpy.ndarray
        The composite signal `c_s`.

    tuple (if verbose=True)
        Tuple containing the composite signal `c_s`,
        the three ZFF signals `y0`, `y1`, `y2`,
        as well as the `gci` locations.
    """

    # Create ZFF signals, from z1 to z4 by modifying trend appropriately
    y0, y1, y2, gci = a_zff_norm_sig(s, fs)

    # Sum to create the composite signal, and normalize
    c_s = y0 + y1 + y2
    c_s /= np.max(np.abs(c_s))

    if verbose:
        return c_s, y0, y1, y2, gci
    else:
        return c_s


def zff_vad(s, fs, spectralEntropy=False, thetaScale=1.0):
    """
    ZFF VAD method demarcates voiced regions using
    a composite signal composed of different zero-frequency
    filtered signals by combining the filter outputs together
    to compose a composite signal carrying f0, F1, and F2 related
    information, and then applying a dynamic threshold after
    spectral entropy-based weighting.

    Parameters
    ----------
    s : numpy.ndarray
        Raw speech signal.
    fs : int
        Sampling frequency of `s`.
    verbose : bool

    Returns
    -------
    numpy.ndarray
        The ZFF-based VAD decision boundary.
    """

    _, y0, y1, y2, _ = zff_cs(s, fs, verbose=True)
    runMean_win = 40 * fs // 1000

    # Weigh normalized ZFF and modZFF with their gradients
    d0 = np.multiply(y0, np.array(list(np.diff(y0)) + [0]))
    d1 = np.multiply(y1, np.array(list(np.diff(y1)) + [0]))
    d2 = np.multiply(y2, np.array(list(np.diff(y2)) + [0]))

    # Normalize signals
    d0 /= np.max(np.abs(d0))
    d1 /= np.max(np.abs(d1))
    d2 /= np.max(np.abs(d2))

    # Smooth signals
    r0 = run_mean(np.abs(d0), runMean_win)[0]
    r1 = run_mean(np.abs(d1), runMean_win)[0]
    r2 = run_mean(np.abs(d2), runMean_win)[0]

    # Normalize
    r0 /= np.max(np.abs(r0))
    r1 /= np.max(np.abs(r1))
    r2 /= np.max(np.abs(r2))

    # Sum to create the summed evidences, and normalize
    r = r0 + r1 + r2
    r /= np.max(np.abs(r))

    if spectralEntropy:
        # Derive inverse spectral entropy over a 30ms block
        entropy = getEntropy(s, fs)
        entropy = np.log(entropy) - 1
        entropy = 1 / entropy

        # Resample spectral entropy over total length of the signal
        nA = np.ceil(len(r) / len(entropy)).astype("int")  # len(s)
        cx = np.tile(entropy.T, (nA, 1))
        cx = cx.T.ravel()
        cx = cx[: len(r)]  # len(s)
        cx /= np.max(np.abs(cx))
        cx = run_mean(np.abs(cx), 40 * fs // 1000)[0]

        # Weigh decision surface with inverse spectral entropy
        decisionSurface = np.multiply(cx, r)

    else:
        decisionSurface = r

    # Calculate threshold for entropy weighed decision surface
    theta = create_th(decisionSurface, winT=80 * fs // 1000)

    # Threshold the decision surface with the calculated theta
    voicedRegions = (decisionSurface > (theta * thetaScale)) * 1

    return voicedRegions
