#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # ---------------------------
# # 1. Define CNN model 
# # ---------------------------
# class CNN(nn.Module):
#     def __init__(self, filter1=128, filter2=32, dropout1=0.5, dropout2=0.3, dropout_fc=0.1):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(1, filter1, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(filter1, filter2, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(filter2 * 61, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.dropout1 = nn.Dropout(dropout1)
#         self.dropout2 = nn.Dropout(dropout2)
#         self.dropout_fc = nn.Dropout(dropout_fc)
#         self.batch_norm1 = nn.BatchNorm1d(filter1)
#         self.batch_norm2 = nn.BatchNorm1d(filter2)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
#         x = self.dropout1(x)
#         x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
#         x = self.dropout2(x)
#         x = x.view(-1, self.fc1.in_features)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)
#         return x
    
# # ---------------------------
# # 2. Load trained peak-finding model
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# peak_finding_model = CNN().to(device)
# parent = os.path.dirname(os.getcwd())
# modelPath=parent+'\\utils\\models'
# peak_finding_model.load_state_dict(torch.load(modelPath+"/waveI_CNN.pth", map_location=device))
# # peak_finding_model.load_state_dict(torch.load("../models/waveI_CNN.pth", map_location=device))
# peak_finding_model.eval()

# ---------------------------
# 3. Normalize waveform before inference
# ---------------------------
def interpolate_and_smooth(y, target_length=244):
    x = np.linspace(0, 1, len(y))
    new_x = np.linspace(0, 1, target_length)
    
    if len(y) == target_length:
        final = y
    elif len(y) > target_length:
        interpolated_values = np.interp(new_x, x, y).astype(float)
        final = pd.Series(interpolated_values)
    elif len(y) < target_length:
        cs = CubicSpline(x, y)
        final = cs(new_x)
    return np.array(final, dtype=float)

def normalize_waveform(wave):
    """Standardize then scale to 0–1."""
    scaler1 = StandardScaler()
    zscored = scaler1.fit_transform(wave.reshape(-1, 1)).flatten()
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler2.fit_transform(zscored.reshape(-1, 1)).flatten()
    return normalized

# ---------------------------
# 4. Peak finding with normalization + smoothing
# ---------------------------
def peak_finding(wave):
    """
    Identify waves I–V using fixed latency windows and interval targets.
    Returns peak and trough indices (samples) on the provided waveform (order: I, II, III, IV, V).
    """
    wave = np.asarray(wave, dtype=float)
    if wave.size == 0:
        return np.array([]), np.array([])

    smoothed_waveform = gaussian_filter1d(wave, sigma=1.0)
    sample_period_ms = 10.0 / len(smoothed_waveform)
    time_axis_ms = np.arange(len(smoothed_waveform)) * sample_period_ms

    # Process Wave III first (typically largest amplitude), then remaining waves.
    #from https://www.sciencedirect.com/science/article/pii/S1879729623000534
    wave_windows = [
        {"name": "III", "start": 3.0, "end": 4.0, "mean": 3.95, "sd": 0.27, "cl": 4.48},
        {"name": "I", "start": 1.0, "end": 2.0, "mean": 1.79, "sd": 0.28, "cl": 2.34},
        {"name": "II", "start": 2.0, "end": 3.0, "mean": None, "sd": None, "cl": None},
        {"name": "IV", "start": 4.0, "end": 5.0, "mean": None, "sd": None, "cl": None},
        {"name": "V", "start": 5.0, "end": 6.0, "mean": 5.85, "sd": 0.31, "cl": 6.45},
    ]
    # values from https://www.thebsa.org.uk/wp-content/uploads/2022/09/ABR-post-newborn-and-Adult.pdf
    interval_targets = {
        ("I", "III"): 2.2, ("III", "I"): 2.2,
        ("III", "V"): 1.8, ("V", "III"): 1.8,
        ("I", "V"): 4.0, ("V", "I"): 4.0
    }
    wave_slots = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4}

    amp_range = float(np.max(smoothed_waveform) - np.min(smoothed_waveform))
    prominence = 0.02 * amp_range if amp_range > 0 else None
    all_peaks, _ = find_peaks(smoothed_waveform, prominence=prominence, distance=3)
    all_troughs, _ = find_peaks(-smoothed_waveform, distance=3)

    # Precompute a paired trough and peak–trough amplitude for every detected peak.
    peak_pairs = {}
    for p in all_peaks:
        after_troughs = all_troughs[all_troughs > p]
        trough_idx = None
        if after_troughs.size > 0:
            # Prefer the closest trough within 1.0 ms
            within_mask = time_axis_ms[after_troughs] <= (time_axis_ms[p] + 1.0)
            within = after_troughs[within_mask]
            if within.size > 0:
                trough_idx = int(within[np.argmin(time_axis_ms[within] - time_axis_ms[p])])
            else:
                trough_idx = int(after_troughs[0])
        pt_amp = smoothed_waveform[p] - (smoothed_waveform[trough_idx] if trough_idx is not None else 0.0)
        peak_pairs[p] = {"trough": trough_idx, "pt_amp": pt_amp}

    selected = []
    for spec in wave_windows:
        in_window = (time_axis_ms[all_peaks] >= spec["start"]) & (time_axis_ms[all_peaks] <= spec["end"])
        candidate_peaks = all_peaks[in_window]
        if candidate_peaks.size == 0:
            continue

        # Use mean/SD as a tie-breaker only (do not discard out-of-band peaks)
        use_mean_sd = spec["mean"] is not None and spec["sd"] is not None

        best_idx = None
        best_score = None
        for cand in candidate_peaks:
            cand_lat = time_axis_ms[cand]
            penalty = 0.0
            for prev in selected:
                key = (prev["name"], spec["name"])
                if key in interval_targets:
                    penalty += abs((cand_lat - prev["lat_ms"]) - interval_targets[key])

            # Tie-breaker toward the published mean only if multiple peaks in window
            mean_penalty = 0.0
            if use_mean_sd and candidate_peaks.size > 1:
                mean_penalty = abs(cand_lat - spec["mean"]) * 0.5

            # Use precomputed peak–trough amplitude; Wave III typically largest.
            pt_amplitude = peak_pairs.get(int(cand), {}).get("pt_amp", smoothed_waveform[cand])

            # Amplitude bias: Wave III gets higher weight to start selection there
            amp_weight = 2.0 if spec["name"] == "III" else 1.0
            score = (penalty, mean_penalty, -amp_weight * pt_amplitude)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = cand

        if best_idx is not None:
            selected.append({
                "name": spec["name"],
                "idx": int(best_idx),
                "lat_ms": float(time_axis_ms[best_idx]),
                "window_end": spec["end"],
                "paired_trough": peak_pairs.get(int(best_idx), {}).get("trough")
            })

    selected_peaks = np.array([s["idx"] for s in selected], dtype=float)

    # Closest following trough; prefer precomputed pair, else immediate follower in short window, else to next peak (or end)
    selected_troughs = []
    for i, s in enumerate(selected):
        peak_idx = s["idx"]
        window_end = s["window_end"]
        next_peak_idx = selected[i + 1]["idx"] if i + 1 < len(selected) else None
        next_peak_time = time_axis_ms[next_peak_idx] if next_peak_idx is not None else time_axis_ms[-1]
        short_cutoff = time_axis_ms[peak_idx] + 0.8

        # Try precomputed pair first
        paired_trough = s.get("paired_trough")
        if paired_trough is not None:
            selected_troughs.append(int(paired_trough))
            continue

        # Try short window first
        short_mask = (all_troughs > peak_idx) & (time_axis_ms[all_troughs] <= min(short_cutoff, next_peak_time))
        candidate_troughs = all_troughs[short_mask]

        # Fallback to full window up to next peak
        if candidate_troughs.size == 0:
            cutoff_time = next_peak_time
            mask = (all_troughs > peak_idx) & (time_axis_ms[all_troughs] <= cutoff_time)
            candidate_troughs = all_troughs[mask]
        if candidate_troughs.size == 0:
            continue
        nearest_trough = int(candidate_troughs[np.argmin(time_axis_ms[candidate_troughs] - time_axis_ms[peak_idx])])
        selected_troughs.append(nearest_trough)

    peaks_out = np.full(5, np.nan)
    troughs_out = np.full(5, np.nan)
    for s in selected:
        slot = wave_slots[s["name"]]
        peaks_out[slot] = s["idx"]
    if selected_troughs:
        # Align troughs to wave order based on temporal pairing in selected
        for s, t in zip(selected, selected_troughs):
            slot = wave_slots[s["name"]]
            troughs_out[slot] = t

    return peaks_out, troughs_out

# ---------------------------
# 5. Convert peaks → latency & amplitude (Wave I only)
# ---------------------------
def extract_latency_amplitude(wave):
    peaks, troughs = peak_finding(wave)

    finite_peaks = peaks[np.isfinite(peaks)]
    finite_troughs = troughs[np.isfinite(troughs)]
    if len(finite_peaks) == 0 or len(finite_troughs) == 0:
        return {
            "peak_idx": None,
            "trough_idx": None,
            "latency_ms": None,
            "amplitude_uV": None
        }

    # Take the FIRST peak + its first corresponding trough
    p = int(finite_peaks[0])
    t = int(finite_troughs[0]) if len(finite_troughs) > 0 else None

    latency_ms = p * (10.0 / len(wave))   # 10 ms / waveform samples
    amplitude_uV = wave[p] - wave[t] if t is not None else None

    return {
        "peak_idx": p,
        "trough_idx": t,
        "latency_ms": latency_ms,
        "amplitude_uV": amplitude_uV
    }



# ---------------------------
# 7. Optional visualization
# ---------------------------
def plot_example(fname):
    df = pd.read_csv(fname)
    wave = df.loc[0, '0':'255'].values.astype(float)
    peaks, troughs = peak_finding(wave)
    time_ms = np.linspace(0, 10, len(wave))

    trough_idx = troughs[np.isfinite(troughs)].astype(int)

    plt.plot(time_ms, wave, label="ABR waveform")
    wave_labels = ["I", "II", "III", "IV", "V"]
    for idx, lbl in enumerate(wave_labels):
        if idx >= len(peaks) or not np.isfinite(peaks[idx]):
            continue
        p = int(peaks[idx])
        if 0 <= p < len(wave):
            plt.scatter(time_ms[p], wave[p], c="r")
            plt.text(time_ms[p], wave[p], lbl, color="r", ha="left", va="bottom")
    for t in trough_idx:
        plt.scatter(time_ms[t], wave[t], c="b")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (µV)")
    plt.title(f"{fname} | Peaks & Troughs")
    plt.legend()
    plt.show()
#%%

# ---------------------------
# Load data from csv to dataframe
# ---------------------------


# fileName = '20251120_ABRConfigAscii_ABRA_passOnly.csv'
fileName = '20251120_ABRConfigAscii_ABRA.csv'
df_data_255 = pd.read_csv(fileName)



base_cols = ['IDnumber', 'SGI', 'Rec No.', 'Ear', 'Time Intervals', 'Freq(Hz)', 'Level(dB)']
cols_244 = [str(i) for i in range(244)]

# Load manual peak/trough inputs if provided (cannot be overwritten)
manual_file = Path("manual_wave_parameters_all.csv")
manual_inputs = pd.DataFrame(columns=["IDnumber", "Ear", "p1", "p2", "p3", "p4", "p5", "t1", "t2", "t3", "t4", "t5"])
if manual_file.exists():
    manual_inputs = pd.read_csv(manual_file)
    for col in ["IDnumber", "Ear", "p1", "p2", "p3", "p4", "p5", "t1", "t2", "t3", "t4", "t5"]:
        if col not in manual_inputs.columns:
            manual_inputs[col] = np.nan
    numeric_cols = ["p1", "p2", "p3", "p4", "p5", "t1", "t2", "t3", "t4", "t5"]
    for c in numeric_cols:
        manual_inputs[c] = pd.to_numeric(manual_inputs.get(c, np.nan), errors="coerce")
    manual_inputs["IDnumber_norm"] = manual_inputs["IDnumber"].astype(str).str.strip().str.lower()
    manual_inputs["Ear_norm"] = manual_inputs["Ear"].astype(str).str.strip().str.upper()

# make a single NaN block with the same index, then concat once
nan_block = pd.DataFrame(np.nan, index=df_data_255.index, columns=cols_244)
df_data_244 = pd.concat([df_data_255[base_cols].copy(), nan_block], axis=1)

# Limit processing to a slice of rows (adjust slice to control how many waves to run)
# nslice= 1+2*31  # set to None to process all rows
# row_slice = slice(nslice, nslice+2)
row_slice = slice(0, len(df_data_255))  # process all rows
rows_iter = df_data_255.iloc[row_slice].itertuples() if row_slice else df_data_255.itertuples()
mean_lines = [(1.79, "I"), (3.95, "III"), (5.85, "V")]  # wave I/III/V mean latencies in ms
ref_windows = [
    ("Wave I window", 1.0, 2.0),
    ("Wave III window", 3.0, 4.0),
    ("Wave V window", 5.0, 6.0),
]

for row in rows_iter:
    
    wave_255=df_data_255.loc[row.Index,'0':'255'].values.astype(float)
    wave_244 = interpolate_and_smooth(np.array(wave_255, dtype=float), target_length=244)

    # Apply manual overrides before auto detection; fill missing with auto
    manual_row = pd.DataFrame()
    if not manual_inputs.empty:
        id_norm = str(df_data_255.loc[row.Index, "IDnumber"]).strip().lower()
        ear_norm = str(df_data_255.loc[row.Index, "Ear"]).strip().upper()
        manual_row = manual_inputs[
            (manual_inputs["IDnumber_norm"] == id_norm) &
            (manual_inputs["Ear_norm"] == ear_norm)
        ]

    def ms_to_idx(arr_ms, n_samples):
        idx = np.full_like(arr_ms, np.nan, dtype=float)
        factor = n_samples / 10.0  # 10 ms window over n_samples
        finite = np.isfinite(arr_ms)
        idx[finite] = np.round(arr_ms[finite] * factor)
        idx[finite] = np.clip(idx[finite], 0, n_samples - 1)
        return idx

    manual_peaks = np.full(5, np.nan)
    manual_troughs = np.full(5, np.nan)
    if not manual_row.empty:
        manual_peaks = ms_to_idx(manual_row.iloc[0][['p1','p2','p3','p4','p5']].to_numpy(dtype=float), len(wave_244))
        manual_troughs = ms_to_idx(manual_row.iloc[0][['t1','t2','t3','t4','t5']].to_numpy(dtype=float), len(wave_244))

    # Auto detect everything first
    auto_peaks, auto_troughs = peak_finding(wave_244)
    peaks = auto_peaks.copy()
    troughs = auto_troughs.copy()

    # Pair helpers: nearest following trough / preceding peak on smoothed wave
    smoothed = gaussian_filter1d(wave_244, sigma=1.0)
    all_auto_peaks, _ = find_peaks(smoothed, distance=3)
    all_auto_troughs, _ = find_peaks(-smoothed, distance=3)

    def nearest_following_trough(peak_idx):
        after = all_auto_troughs[all_auto_troughs > peak_idx]
        if after.size == 0:
            return None
        return int(after[np.argmin(after - peak_idx)])

    def nearest_preceding_peak(trough_idx):
        before = all_auto_peaks[all_auto_peaks < trough_idx]
        if before.size == 0:
            return None
        return int(before[np.argmin(trough_idx - before)])

    # Apply manual peaks/troughs, then fill missing counterpart
    for i in range(5):
        mp = manual_peaks[i]
        mt = manual_troughs[i]
        if np.isfinite(mp):
            peaks[i] = mp
            if not np.isfinite(mt):
                fill_t = nearest_following_trough(int(mp))
                if fill_t is not None:
                    troughs[i] = fill_t
        if np.isfinite(mt):
            troughs[i] = mt
            if not np.isfinite(mp):
                fill_p = nearest_preceding_peak(int(mt))
                if fill_p is not None:
                    peaks[i] = fill_p


# store
    for i, key in enumerate(['p1','p2','p3','p4','p5']):
        val = peaks[i] if i < len(peaks) else np.nan
        df_data_244.loc[row.Index, key] = int(val) if np.isfinite(val) else None
    for i, key in enumerate(['t1','t2','t3','t4','t5']):
        val = troughs[i] if i < len(troughs) else np.nan
        df_data_244.loc[row.Index, key] = int(val) if np.isfinite(val) else None

    #conver peaks and troughts fo the 255 scale
    for i, key in enumerate(['p1','p2','p3','p4','p5']):
        val = peaks[i] if i < len(peaks) else np.nan
        peaks255 = int(round(val * (255 / 244))) if np.isfinite(val) else None
        df_data_255.loc[row.Index, key] = peaks255  
    for i, key in enumerate(['t1','t2','t3','t4','t5']):
        val = troughs[i] if i < len(troughs) else np.nan
        troughs255 = int(round(val * (255 / 244))) if np.isfinite(val) else None
        df_data_255.loc[row.Index, key] = troughs255  


    for i, key in enumerate(['w1','w2','w3','w4','w5']):

        raw_p = peaks[i] if i < len(peaks) else np.nan
        raw_t = troughs[i] if i < len(troughs) else np.nan
        p = int(raw_p) if np.isfinite(raw_p) else None
        t = int(raw_t) if np.isfinite(raw_t) else None
        p255= int(round(p * (255 / 244))) if p is not None else None
        t255= int(round(t * (255 / 244))) if t is not None else None
        

        # latency_ms = (p * (10.0 / 244)) if (p not None) else None   # 10 ms / 244 samples
        latency_ms = float(p) * (10.0 / 244) if p is not None else None
        amplitude_uV = wave_244[p] - wave_244[t] if t is not None else None
        df_data_244.loc[row.Index, key+'_latency_ms'] = latency_ms
        df_data_244.loc[row.Index, key+'_amplitude_uV'] = amplitude_uV
        #convert to 255 scale for df_data_255

        latency_ms_255 = float(p255) * (10.0 / 255) if p255 is not None else None
        amplitude_uV_255 = wave_255[p255] - wave_255[t255] if t255 is not None else None
        df_data_255.loc[row.Index, key+'_latency_ms'] = latency_ms_255
        df_data_255.loc[row.Index, key+'_amplitude_uV'] = amplitude_uV_255

    # store the resampled wave back into df_data_244
    df_data_244.loc[row.Index, cols_244] = wave_244

    manual_used = np.any(np.isfinite(manual_peaks)) or np.any(np.isfinite(manual_troughs))
    source_tag = "manual+auto" if manual_used else "auto"
    print(f"Row {row.Index} [{source_tag}] Peaks: {peaks}, Troughs: {troughs}")

#save dataframe in csv includ date in filename
today_date = pd.Timestamp.now().strftime('%Y%m%d')
# df_data_244.to_csv(f'{today_date}_ABRConfigAscii_ABRA_Peaks_244.csv', index=False)
df_data_255.to_csv(f'{today_date}_ABRConfigAscii_ABRA_Peaks_255.csv', index=False)


##%% plot the waves with peaks and troughs from the dataframe
# for row in df_data_244.iloc[0:2].itertuples():    
for row in df_data_244.iloc[row_slice].itertuples(): 
    one_wave=df_data_244.loc[row.Index,'0':'243'].values.astype(float)
    time = np.linspace(0, 10, len(one_wave))  # Assuming 10 ms duration

    wave = df_data_244.loc[row.Index,'0':'243'].values.astype(float)
    time_ms = np.linspace(0, 10, len(wave))
    peaks = pd.to_numeric(df_data_244.loc[row.Index, 'p1':'p5'], errors='coerce').to_numpy(dtype=float)
    troughs = pd.to_numeric(df_data_244.loc[row.Index, 't1':'t5'], errors='coerce').to_numpy(dtype=float)
    ear=df_data_244.loc[row.Index,'Ear']
    id=df_data_244.loc[row.Index,'IDnumber']
    peaks_ms = np.where(np.isfinite(peaks), peaks * (10.0 / 244), np.nan)
    troughs_ms = np.where(np.isfinite(troughs), troughs * (10.0 / 244), np.nan)
    print(f" ID: {id} | Ear: {ear} | Peaks: {peaks_ms} | Troughs: {troughs_ms}")

    # Keep wave identity even if some are missing (avoid shifting labels).
    peak_idx = peaks[np.isfinite(peaks)].astype(int)
    trough_idx = troughs[np.isfinite(troughs)].astype(int)
    peak_idx = peak_idx[peak_idx < len(wave)]
    trough_idx = trough_idx[trough_idx < len(wave)]

    plt.figure(figsize=(10*1.4, 4*1.4))
    plt.plot(time_ms, wave, label='ABR Waveform')
    plt.scatter(time_ms[peak_idx], wave[peak_idx], color='red', label='Peaks')
    plt.scatter(time_ms[trough_idx], wave[trough_idx], color='blue', label='Troughs')
    for idx, (label, start, end) in enumerate(ref_windows):
        plt.axvspan(start, end, color='orange', alpha=0.12,
                    label='Reference windows (I/III/V)' if idx == 0 else None)
        plt.text((start + end) / 2, np.max(wave) * 0.9, label,
                 ha='center', va='top', fontsize=9, color='sienna', alpha=0.8)
    wave_labels = ["I", "II", "III", "IV", "V"]
    for idx, lbl in enumerate(wave_labels):
        if idx >= len(peaks) or not np.isfinite(peaks[idx]):
            continue
        p = int(peaks[idx])
        if 0 <= p < len(wave):
            plt.text(time_ms[p], wave[p], lbl, color="red", ha="left", va="bottom")
    for idx, (t, lbl) in enumerate(mean_lines):
        plt.axvline(t, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                    label='Wave mean (I/III/V)' if idx == 0 else None)
    plt.title(f'ABR Waveform for {id} - {ear} Ear')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (µV)')
    plt.xlim(0, 10)
    # plt.legend()
    
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 10, 0.5)
    minor_ticks = np.arange(0, 10, 0.1)
    


    plt.xticks(major_ticks)
    plt.xticks(minor_ticks, minor=True)
    
    
    plt.grid()
    plt.show()

#%% regenerate the plots and save to a common pdf file
from matplotlib.backends.backend_pdf import PdfPages

# intro_page_lines = [
#     "ABR peak-finding overview (10 ms waveform, 256 samples).",
#     "- Smooth waveform with Gaussian sigma=1.0; compute time axis in ms.",
#     "- Detect peaks with find_peaks; prominence = 0.02 * amplitude range; distance = 3 samples.",
#     "- Pair each peak with the nearest following trough (prefer within 1 ms) to measure peak-to-trough amplitude.",
#     "- Search waves in order III, I, II, IV, V using fixed windows (ms): I 1-2, II 2-3, III 3-4, IV 4-5, V 5-6.",
#     "- Score using inter-wave interval targets (I-III 2.2 ms, III-V 1.8 ms, I-V 4.0 ms) plus an amplitude bias favoring Wave III.",
#     "- Mean/SD latencies: I 1.79+/-0.28 ms, III 3.95+/-0.27 ms, V 5.85+/-0.31 ms; used only as tie-breakers when multiple peaks sit in-window.",
#     "- Troughs: prefer the precomputed pair; otherwise choose the nearest following trough before the next peak.",
#     "References:",
#     "- Latency windows: S. Kerneis, E. Caillaud, D. Bakhos, Eur Ann Otorhinolaryngol Head Neck Dis 140(4), 2023, https://doi.org/10.1016/j.anorl.2023.04.003.",
#     "- Mean/SD latencies: British Society of Audiology, ABR post-newborn and Adult (2022), https://www.thebsa.org.uk/wp-content/uploads/2022/09/ABR-post-newborn-and-Adult.pdf."
# ]

intro_page_lines = [
    "ABR peak-finding overview (10 ms waveform, 256 samples).",
    "- Smooth waveform with Gaussian sigma=1.0; compute time axis in ms.",
    "- Detect peaks with find_peaks; prominence = 0.02 * amplitude range; distance = 3 samples.",
    "- Pair each peak with the nearest following trough (prefer within 1 ms) to measure peak-to-trough amplitude.",
    "- Manual overrides: if manual_wave_parameters_all.csv has entries for ID+Ear, convert those latencies (ms) to indices and apply before auto detection; blanks fall back to auto.",
    "- Search waves in order III, I, II, IV, V using fixed windows (ms): I 1-2, II 2-3, III 3-4, IV 4-5, V 5-6.",
    "- Score using inter-wave interval targets (I-III 2.2 ms, III-V 1.8 ms, I-V 4.0 ms) plus an amplitude bias favoring Wave III.",
    "- Mean/SD latencies: I 1.79+/-0.28 ms, III 3.95+/-0.27 ms, V 5.85+/-0.31 ms; used only as tie-breakers when multiple peaks sit in-window.",
    "- Troughs: prefer the precomputed pair; otherwise choose the nearest following trough before the next peak.",
    "References:",
    "- Latency windows: S. Kerneis, E. Caillaud, D. Bakhos, Eur Ann Otorhinolaryngol Head Neck Dis 140(4), 2023, https://doi.org/10.1016/j.anorl.2023.04.003.",
    "- Mean/SD latencies: British Society of Audiology, ABR post-newborn and Adult (2022), https://www.thebsa.org.uk/wp-content/uploads/2022/09/ABR-post-newborn-and-Adult.pdf."
]

def add_intro_page(pdf_handle):
    import textwrap
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("ABR Peak-Finding Summary", fontsize=14, y=0.98)
    y = 0.92
    wrap_width = 95
    for line in intro_page_lines:
        wrapped = textwrap.wrap(line, width=wrap_width) or [line]
        for w in wrapped:
            fig.text(0.06, y, w, fontsize=11, ha="left", va="top")
            y -= 0.032
        y -= 0.01  # small gap between bullets
    fig.text(0.06, y - 0.015, f"Generated {today_date}", fontsize=10, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_handle.savefig(fig)
    plt.close(fig)


with PdfPages(f'{today_date}_ABR_Waveforms_Peaks_Detection.pdf') as pdf:
    add_intro_page(pdf)
    for row in df_data_244.iloc[row_slice].itertuples(): 
        one_wave=df_data_244.loc[row.Index,'0':'243'].values.astype(float)
        time = np.linspace(0, 10, len(one_wave))  # Assuming 10 ms duration

        wave = df_data_244.loc[row.Index,'0':'243'].values.astype(float)
        time_ms = np.linspace(0, 10, len(wave))
        peaks = pd.to_numeric(df_data_244.loc[row.Index, 'p1':'p5'], errors='coerce').to_numpy(dtype=float)
        troughs = pd.to_numeric(df_data_244.loc[row.Index, 't1':'t5'], errors='coerce').to_numpy(dtype=float)
        ear=df_data_244.loc[row.Index,'Ear']
        id=df_data_244.loc[row.Index,'IDnumber']

        peak_idx = peaks[np.isfinite(peaks)].astype(int)
        trough_idx = troughs[np.isfinite(troughs)].astype(int)
        peak_idx = peak_idx[peak_idx < len(wave)]
        trough_idx = trough_idx[trough_idx < len(wave)]

        plt.figure(figsize=(10, 4))
        plt.plot(time_ms, wave, label='ABR Waveform')
        plt.scatter(time_ms[peak_idx], wave[peak_idx], color='red', label='Peaks')
        plt.scatter(time_ms[trough_idx], wave[trough_idx], color='blue', label='Troughs')
        for idx, (label, start, end) in enumerate(ref_windows):
            plt.axvspan(start, end, color='orange', alpha=0.12,
                        label='Reference windows (I/III/V)' if idx == 0 else None)
            plt.text((start + end) / 2, np.max(wave) * 0.9, label,
                     ha='center', va='top', fontsize=9, color='sienna', alpha=0.8)
        wave_labels = ["I", "II", "III", "IV", "V"]
        for idx, lbl in enumerate(wave_labels):
            if idx >= len(peaks) or not np.isfinite(peaks[idx]):
                continue
            p = int(peaks[idx])
            if 0 <= p < len(wave):
                plt.text(time_ms[p], wave[p], lbl, color="red", ha="left", va="bottom")
        for idx, (t, lbl) in enumerate(mean_lines):
            plt.axvline(t, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                        label='Wave mean (I/III/V)' if idx == 0 else None)
        plt.title(f'ABR Waveform for {id} - {ear} Ear')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.xlim(0, 10)
        plt.legend()
        plt.grid()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


# %%
#%% regenerate the plots and save to a common pdf file (multiple plots per page, keep same ID together)
from matplotlib.backends.backend_pdf import PdfPages
import math
from collections import OrderedDict

plots_per_page = 4  # adjust if you want more/less per page

# group rows by ID (preserve order) so a single ID is never split across pages
grouped_rows = OrderedDict()
for r in df_data_244.iloc[0:63].itertuples():
    key = df_data_244.loc[r.Index, 'IDnumber']
    grouped_rows.setdefault(key, []).append(r)


def render_page(rows_chunk, pdf_handle):
    if not rows_chunk:
        return
    nplots = len(rows_chunk)
    ncols = 2
    nrows = math.ceil(nplots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, row in zip(axes_flat, rows_chunk):
        wave = df_data_244.loc[row.Index, '0':'243'].values.astype(float)
        time_ms = np.linspace(0, 10, len(wave))
        peaks = pd.to_numeric(df_data_244.loc[row.Index, 'p1':'p5'], errors='coerce').to_numpy(dtype=float)
        troughs = pd.to_numeric(df_data_244.loc[row.Index, 't1':'t5'], errors='coerce').to_numpy(dtype=float)
        ear = df_data_244.loc[row.Index, 'Ear']
        id_val = df_data_244.loc[row.Index, 'IDnumber']

        peak_idx = peaks[np.isfinite(peaks)].astype(int)
        trough_idx = troughs[np.isfinite(troughs)].astype(int)
        peak_idx = peak_idx[peak_idx < len(wave)]
        trough_idx = trough_idx[trough_idx < len(wave)]

        ax.plot(time_ms, wave, label='ABR Waveform')
        ax.scatter(time_ms[peak_idx], wave[peak_idx], color='red', label='Peaks')
        ax.scatter(time_ms[trough_idx], wave[trough_idx], color='blue', label='Troughs')
        for idx, (label, start, end) in enumerate(ref_windows):
            ax.axvspan(start, end, color='orange', alpha=0.12,
                       label='Reference windows (I/III/V)' if idx == 0 else None)
            ax.text((start + end) / 2, np.max(wave) * 0.9, label[0:-6],
                    ha='center', va='top', fontsize=9, color='sienna', alpha=0.8)
        wave_labels = ["I", "II", "III", "IV", "V"]
        for idx, lbl in enumerate(wave_labels):
            if idx >= len(peaks) or not np.isfinite(peaks[idx]):
                continue
            p = int(peaks[idx])
            if 0 <= p < len(wave):
                ax.text(time_ms[p], wave[p], lbl, color="red", ha="left", va="bottom")
        for idx, (t, lbl) in enumerate(mean_lines):
            ax.axvline(t, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                       label='Wave mean (I/III/V)' if idx == 0 else None)
        ax.set_title(f'ABR Waveform for {id_val} - {ear} Ear')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (µV)')
        ax.set_xlim(0, 10)
        ax.grid()

    # axes_flat[0].legend()
    for ax in axes_flat[nplots:]:
        ax.set_visible(False)

    fig.tight_layout()
    pdf_handle.savefig(fig)
    plt.close(fig)


with PdfPages(f'{today_date}_ABR_Waveforms_Peaks_Detection.pdf') as pdf:
    add_intro_page(pdf)
    current = []
    for _, rows_for_id in grouped_rows.items():
        # if a single ID exceeds page capacity, flush current page then chunk the big group
        if len(rows_for_id) > plots_per_page:
            if current:
                render_page(current, pdf)
                current = []
            for start in range(0, len(rows_for_id), plots_per_page):
                render_page(rows_for_id[start:start + plots_per_page], pdf)
        else:
            if current and len(current) + len(rows_for_id) > plots_per_page:
                render_page(current, pdf)
                current = []
            current.extend(rows_for_id)
    if current:
        render_page(current, pdf)
