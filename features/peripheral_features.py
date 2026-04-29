"""
Peripheral physiological feature extraction: PPG (BVP) and GSR (EDA).

PPG → 10 HRV features (time-domain + frequency-domain + morphological)
GSR → 8 EDA features (tonic SCL + phasic SCR components)

All features are extracted per trial from 60-second recordings at 128 Hz.
"""
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.ndimage import uniform_filter1d

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SFREQ, N_PPG_FEATURES, N_GSR_FEATURES


# ─── PPG / HRV features ───────────────────────────────────────────────────────

def _butter_bandpass(low: float, high: float, fs: int = SFREQ, order: int = 3):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


_PPG_SOS = _butter_bandpass(0.5, 8.0)   # typical PPG / BVP band


def extract_ppg_features(ppg: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    """
    Extract HRV features from a single PPG/BVP trial.

    Features (10):
      0  mean_hr         — mean heart rate (BPM)
      1  sdnn            — std of NN intervals (ms)
      2  rmssd           — root mean square of successive differences (ms)
      3  pnn50           — % of successive diffs > 50 ms
      4  lf_power        — LF power [0.04–0.15 Hz] (log-scaled)
      5  hf_power        — HF power [0.15–0.4  Hz] (log-scaled)
      6  lf_hf_ratio     — LF/HF ratio
      7  mean_amp        — mean peak-to-peak amplitude
      8  std_amp         — std of peak-to-peak amplitudes
      9  ibi_cv          — coefficient of variation of IBI

    Args:
        ppg: (n_samples,) raw PPG signal
        fs: sampling rate

    Returns:
        features: (10,) float32
    """
    # 1. Filter
    ppg_f = sosfiltfilt(_PPG_SOS, ppg)

    # Normalize to [-1, 1]
    p_min, p_max = ppg_f.min(), ppg_f.max()
    if p_max - p_min > 1e-8:
        ppg_f = 2 * (ppg_f - p_min) / (p_max - p_min) - 1

    # 2. Detect peaks (R-peaks surrogate)
    min_dist = int(0.4 * fs)   # ≥ 40 BPM
    max_dist = int(1.5 * fs)   # ≤ 150 BPM (40 BPM floor)
    peaks, props = find_peaks(ppg_f, distance=min_dist, prominence=0.1)

    feats = np.zeros(N_PPG_FEATURES, dtype=np.float32)

    if len(peaks) < 3:
        # Not enough peaks — return zeros (signal too noisy)
        return feats

    # 3. IBI (inter-beat intervals) in milliseconds
    ibi = np.diff(peaks) / fs * 1000.0    # ms

    # Filter physiologically implausible IBIs (300–2000 ms → 30–200 BPM)
    valid = (ibi > 300) & (ibi < 2000)
    if valid.sum() < 2:
        return feats
    ibi = ibi[valid]

    # Time-domain HRV
    mean_hr  = 60000.0 / ibi.mean()
    sdnn     = ibi.std(ddof=1)
    diffs    = np.diff(ibi)
    rmssd    = np.sqrt(np.mean(diffs ** 2))
    pnn50    = 100.0 * np.mean(np.abs(diffs) > 50)
    ibi_cv   = sdnn / (ibi.mean() + 1e-8)

    # Frequency-domain HRV via Welch on evenly-resampled IBI series
    # Resample IBI to 4 Hz via linear interpolation
    t_ibi = np.cumsum(ibi) / 1000.0   # cumulative time in seconds
    t_uniform = np.arange(t_ibi[0], t_ibi[-1], 0.25)
    if len(t_uniform) < 8:
        lf_power, hf_power, lf_hf = 0.0, 0.0, 0.0
    else:
        ibi_uniform = np.interp(t_uniform, t_ibi, ibi)
        freqs, psd = welch(ibi_uniform, fs=4.0, nperseg=min(len(ibi_uniform), 64))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power  = float(np.log1p(psd[lf_mask].sum()))
        hf_power  = float(np.log1p(psd[hf_mask].sum()))
        lf_hf     = lf_power / (hf_power + 1e-8)

    # Peak amplitude
    amps = ppg_f[peaks]
    mean_amp = float(amps.mean())
    std_amp  = float(amps.std(ddof=1)) if len(amps) > 1 else 0.0

    feats[:] = [mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf,
                mean_amp, std_amp, ibi_cv]
    return feats.astype(np.float32)


# ─── GSR / EDA features ───────────────────────────────────────────────────────

def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    return uniform_filter1d(x.astype(np.float64), size=win)


def extract_gsr_features(gsr: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    """
    Extract EDA features using a simple tonic/phasic decomposition.

    Tonic (SCL): slow-varying baseline via large moving average
    Phasic (SCR): residual = EDA - SCL

    Features (8):
      0  scl_mean      — mean tonic level
      1  scl_std       — std of tonic level
      2  eda_mean      — mean raw EDA
      3  eda_std       — std raw EDA
      4  scr_n_peaks   — number of SCR peaks
      5  scr_mean_amp  — mean SCR peak amplitude
      6  scr_rise_rate — mean SCR rise rate (amplitude / rise time)
      7  eda_slope     — linear trend slope of EDA signal

    Args:
        gsr: (n_samples,) raw EDA signal
        fs: sampling rate

    Returns:
        features: (8,) float32
    """
    gsr = gsr.astype(np.float64)
    feats = np.zeros(N_GSR_FEATURES, dtype=np.float32)

    # Normalize to µS-range if raw values are large
    if gsr.std() > 1e3:
        gsr = (gsr - gsr.mean()) / (gsr.std() + 1e-8)

    # Tonic: moving average over 4 seconds
    tonic_win = int(4 * fs)
    scl = _smooth(gsr, tonic_win)
    phasic = gsr - scl

    # SCR peaks in phasic component
    min_dist = int(0.5 * fs)
    peaks, _ = find_peaks(phasic, distance=min_dist, prominence=0.01)

    n_peaks    = len(peaks)
    mean_amp   = float(phasic[peaks].mean()) if n_peaks > 0 else 0.0
    rise_rates = []
    for pk in peaks:
        onset = max(0, pk - int(0.5 * fs))
        rise = phasic[pk] - phasic[onset]
        time = (pk - onset) / fs + 1e-8
        rise_rates.append(rise / time)
    mean_rise = float(np.mean(rise_rates)) if rise_rates else 0.0

    # EDA slope
    t = np.arange(len(gsr))
    slope = float(np.polyfit(t, gsr, 1)[0])

    feats[:] = [
        scl.mean(), scl.std(ddof=1),
        gsr.mean(), gsr.std(ddof=1),
        n_peaks, mean_amp, mean_rise, slope,
    ]
    return feats.astype(np.float32)


# ─── Batch extraction ────────────────────────────────────────────────────────

def extract_ppg_subject(
    ppg_trials: np.ndarray,
    n_windows: int | None = None,
) -> np.ndarray:
    """
    (n_trials, n_samples) -> (n_trials * n_windows, 10)

    If n_windows given, each trial's feature vector is repeated n_windows times
    to align with the window-based EEG features.
    """
    per_trial = np.stack([extract_ppg_features(ppg_trials[i])
                          for i in range(len(ppg_trials))], axis=0)  # (n_trials, 10)
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)   # (n_trials*n_windows, 10)


def extract_gsr_subject(
    gsr_trials: np.ndarray,
    n_windows: int | None = None,
) -> np.ndarray:
    """
    (n_trials, n_samples) -> (n_trials * n_windows, 8)

    If n_windows given, each trial's feature vector is repeated n_windows times.
    """
    per_trial = np.stack([extract_gsr_features(gsr_trials[i])
                          for i in range(len(gsr_trials))], axis=0)  # (n_trials, 8)
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)   # (n_trials*n_windows, 8)
