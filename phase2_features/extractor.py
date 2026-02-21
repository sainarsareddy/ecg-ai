import neurokit2 as nk
import numpy as np

SAMPLING_RATE = 360

def extract_features(ecg_signal: list) -> dict:
    signal = np.array(ecg_signal, dtype=float)

    # Step 1 — Signal quality
    try:
        quality = nk.ecg_quality(signal, sampling_rate=SAMPLING_RATE)
        quality_score = float(np.mean(quality))
    except:
        quality_score = 0.0

    if quality_score < 0.05:
        return {
            "signal_quality": round(quality_score, 3),
            "error": "Signal quality too low. Please re-record ECG."
        }

    # Step 2 — Clean signal
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=SAMPLING_RATE)
    except:
        cleaned = signal

    # Step 3 — Process and extract
    try:
        signals, info = nk.ecg_process(cleaned, sampling_rate=SAMPLING_RATE)

        # Heart Rate
        hr_values = signals["ECG_Rate"].dropna().values
        heart_rate = float(np.mean(hr_values)) if len(hr_values) > 0 else None

        # RR Interval
        r_peaks = np.array(info["ECG_R_Peaks"])
        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks) / SAMPLING_RATE * 1000
            rr_mean = float(np.mean(rr_intervals))
            rr_std = float(np.std(rr_intervals))
            rr_irregular = bool(rr_std > 50)
        else:
            rr_mean = rr_std = None
            rr_irregular = None

        # QRS Duration
        qrs_onsets = np.array(info.get("ECG_Q_Peaks", []))
        qrs_offsets = np.array(info.get("ECG_S_Peaks", []))
        if len(qrs_onsets) > 0 and len(qrs_offsets) > 0:
            min_len = min(len(qrs_onsets), len(qrs_offsets))
            qrs_duration = float(np.mean(
                (qrs_offsets[:min_len] - qrs_onsets[:min_len]) / SAMPLING_RATE * 1000
            ))
        else:
            qrs_duration = None

        # PR Interval
        p_peaks = np.array(info.get("ECG_P_Peaks", []))
        if len(p_peaks) > 0 and len(r_peaks) > 0:
            min_len = min(len(p_peaks), len(r_peaks))
            pr_interval = float(np.mean(
                (r_peaks[:min_len] - p_peaks[:min_len]) / SAMPLING_RATE * 1000
            ))
        else:
            pr_interval = None

        # QT Interval
        t_offsets = np.array(info.get("ECG_T_Offsets", []))
        q_peaks = np.array(info.get("ECG_Q_Peaks", []))
        if len(t_offsets) > 0 and len(q_peaks) > 0:
            min_len = min(len(t_offsets), len(q_peaks))
            qt_interval = float(np.mean(
                (t_offsets[:min_len] - q_peaks[:min_len]) / SAMPLING_RATE * 1000
            ))
        else:
            qt_interval = None

        return {
            "signal_quality": round(quality_score, 3),
            "heart_rate": round(heart_rate, 1) if heart_rate is not None else None,
            "rr_interval_ms": round(rr_mean, 1) if rr_mean is not None else None,
            "rr_std_ms": round(rr_std, 1) if rr_std is not None else None,
            "rr_irregular": rr_irregular,
            "qrs_duration_ms": round(qrs_duration, 1) if qrs_duration is not None else None,
            "pr_interval_ms": round(pr_interval, 1) if pr_interval is not None else None,
            "qt_interval_ms": round(qt_interval, 1) if qt_interval is not None and not np.isnan(qt_interval) else None,
        }

    except Exception as e:
        return {
            "signal_quality": round(quality_score, 3),
            "error": f"Feature extraction failed: {str(e)}"
        }