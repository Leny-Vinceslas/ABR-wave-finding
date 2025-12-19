# ABR Peak-Finding Algorithm

## Overview
- Goal: identify ABR waves I–V on a 10 ms waveform sampled to 244 points.
- Preprocess: Gaussian smooth (`sigma=1.0`) to reduce noise while preserving peaks.
- Time axis: `sample_period_ms = 10 / N`; time in ms for each sample.
- Peak detection: `scipy.signal.find_peaks` on the smoothed waveform with `prominence = 0.02 * (max - min)` and `distance=3`; troughs via `find_peaks` on the inverted signal.
- Peak–trough pairing: for every detected peak, pick the nearest following trough (prefer within 1 ms); store peak–trough amplitude.
- Wave search windows (ms): I: 1–2, II: 2–3, III: 3–4, IV: 4–5, V: 5–6 (from Kerneis et al., 2023).
- Selection order: start with Wave III (largest expected amplitude), then I, II, IV, V.
- Scoring inside each window:
  - Enforce only the window as the hard constraint.
  - Interval penalties: prefer peaks whose latencies fit published inter-wave intervals (I–III ≈2.2 ms, III–V ≈1.8 ms, I–V ≈4.0 ms).
  - Amplitude bias: Wave III is weighted higher to encourage selection there first.
  - Mean/SD latencies (for I/III/V) used only as tie-breakers when multiple peaks exist in the window; peaks are no longer rejected for falling outside mean ± 2 SD.
- Trough selection: for each chosen peak, use its precomputed pair; otherwise pick the nearest following trough before the next peak.

## Key Parameters
- Smoothing: Gaussian `sigma=1.0`.
- Peak prominence: `0.02 * amplitude_range`.
- Min peak distance: 3 samples.
- Wave windows (ms): I 1–2, II 2–3, III 3–4, IV 4–5, V 5–6.
- Mean/SD latencies (ms): I 1.79 ± 0.28, III 3.95 ± 0.27, V 5.85 ± 0.31.
- Interval targets (ms): I–III 2.2, III–V 1.8, I–V 4.0.

## Input File
- Expected input: ABRA CSV export.
- Layout: one waveform per row with metadata columns `IDnumber`, `SGI`, `Rec No.`, `Ear`, `Time Intervals`, `Freq(Hz)`, `Level(dB)`.
- Waveform samples: 256 numeric columns labeled `0`–`255` holding the 10 ms ABR trace (≈39 µs per sample); script resamples to 244 points internally.
- Output: writes detected peaks/troughs and amplitudes back to the dataframe and saves `<YYYYMMDD>_ABRConfigAscii_ABRA_Peaks_255.csv` plus a PDF of plotted traces.

## Manual Overrides
- File: `manual_wave_parameters_all.csv`.
- Contents: rows keyed by `IDnumber` and `Ear` with optional peak/trough latencies in ms (`p1`–`p5`, `t1`–`t5`).
- Behavior: when a matching row exists, manual peaks/troughs are converted from ms to indices and applied before auto detection. Any waves left blank fall back to auto detection so you can supply only the fixes you trust.

## References
- Latency windows: S. Kerneis, E. Caillaud, D. Bakhos, “Auditory brainstem response: Key parameters for good-quality recording,” *European Annals of Otorhinolaryngology, Head and Neck Diseases*, 140(4), 2023, 181–185, ISSN 1879-7296, https://doi.org/10.1016/j.anorl.2023.04.003.
- Mean/SD latencies: British Society of Audiology, “ABR post-newborn and Adult,” 2022. https://www.thebsa.org.uk/wp-content/uploads/2022/09/ABR-post-newborn-and-Adult.pdf
