"""Quench-detection benchmark using synthetic voltage signals.

Run:
    py quench_detection_simulator.py

Needs NumPy and Matplotlib. No real CERN data is used.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# Main settings. Change these for a single-run experiment.
DT = 1e-4
T_END = 0.100
T_QUENCH = 0.030
RISE_RATE = 4.0
NOISE_STD = 0.020
THRESHOLD = 0.100
VALIDATION_MS = 10.0
SPIKE_TIME = 0.045
SPIKE_HEIGHT = 0.130
SPIKE_WIDTH_SAMPLES = 3
N_TRIALS = 150
SEED = 7

HERE = Path(__file__).resolve().parent
OUT = HERE / "quench_results"
OUT.mkdir(exist_ok=True)

TIME = np.arange(0.0, T_END, DT)
TIME_MS = TIME * 1000.0


def make_quench_signal(time_s=TIME, onset_s=T_QUENCH, rise_rate=RISE_RATE):
    """Make a simple voltage ramp that starts at the chosen onset time."""
    signal = np.zeros_like(time_s)
    after_onset = time_s >= onset_s
    signal[after_onset] = rise_rate * (time_s[after_onset] - onset_s)
    return signal


def add_noise(signal, std_v, rng):
    return signal + rng.normal(0.0, std_v, size=signal.size)


def add_spike(signal, time_s=TIME, at_s=SPIKE_TIME,
              height_v=SPIKE_HEIGHT, width_samples=SPIKE_WIDTH_SAMPLES):
    """Add a short rectangular voltage spike to a copy of the signal."""
    result = signal.copy()
    centre = int(np.argmin(np.abs(time_s - at_s)))
    start = max(0, centre - width_samples // 2)
    stop = min(result.size, start + width_samples)
    result[start:stop] += height_v
    return result


def first_threshold_crossing(signal, threshold_v):
    """Return the first crossing index, or None if there is no crossing."""
    hits = np.flatnonzero(signal >= threshold_v)
    return int(hits[0]) if hits.size else None


def persistent_threshold_crossing(signal, threshold_v, validation_samples):
    """Trigger only after consecutive samples stay above the threshold."""
    n = max(1, int(validation_samples))
    if n > signal.size:
        return None

    above = (signal >= threshold_v).astype(np.int32)
    cumulative = np.concatenate(([0], np.cumsum(above)))
    window_counts = cumulative[n:] - cumulative[:-n]
    hits = np.flatnonzero(window_counts == n)

    if hits.size == 0:
        return None
    return int(hits[0] + n - 1)


def evaluate_detectors(signal, threshold_v, validation_samples):
    simple = first_threshold_crossing(signal, threshold_v)
    persistent = persistent_threshold_crossing(
        signal, threshold_v, validation_samples
    )
    return simple, persistent


def make_trial(rng, quench, noise_std, spike=False):
    base = make_quench_signal() if quench else np.zeros_like(TIME)
    measured = add_noise(base, noise_std, rng)
    if spike:
        measured = add_spike(measured)
    return base, measured


def percent(count, total):
    return 100.0 * count / total if total else np.nan


def trial_summary(quench, spike, threshold_v, validation_samples,
                  noise_std, trials, seed):
    """Run repeated trials and return detection and false-alarm statistics."""
    rng = np.random.default_rng(seed)
    simple_post_onset = 0
    persistent_post_onset = 0
    simple_early = 0
    persistent_early = 0
    simple_false = 0
    persistent_false = 0
    simple_latency = []
    persistent_latency = []

    for _ in range(trials):
        _, measured = make_trial(rng, quench, noise_std, spike=spike)
        simple_idx, persistent_idx = evaluate_detectors(
            measured, threshold_v, validation_samples
        )

        if quench:
            for idx, detector in ((simple_idx, "simple"),
                                  (persistent_idx, "persistent")):
                if idx is None:
                    continue
                if TIME[idx] < T_QUENCH:
                    if detector == "simple":
                        simple_early += 1
                    else:
                        persistent_early += 1
                else:
                    latency_ms = (TIME[idx] - T_QUENCH) * 1000.0
                    if detector == "simple":
                        simple_post_onset += 1
                        simple_latency.append(latency_ms)
                    else:
                        persistent_post_onset += 1
                        persistent_latency.append(latency_ms)
        else:
            simple_false += simple_idx is not None
            persistent_false += persistent_idx is not None

    scenario = "quench" if quench else ("no_quench_spike" if spike else "no_quench_noise")
    return {
        "scenario": scenario,
        "noise_std_mV": round(noise_std * 1000.0, 4),
        "threshold_mV": round(threshold_v * 1000.0, 4),
        "validation_ms": round(validation_samples * DT * 1000.0, 4),
        "trials": trials,
        "simple_post_onset_trigger_pct": percent(simple_post_onset, trials),
        "persistent_post_onset_trigger_pct": percent(persistent_post_onset, trials),
        "simple_pre_onset_trigger_pct": percent(simple_early, trials) if quench else np.nan,
        "persistent_pre_onset_trigger_pct": percent(persistent_early, trials) if quench else np.nan,
        "simple_false_alarm_pct": percent(simple_false, trials) if not quench else np.nan,
        "persistent_false_alarm_pct": percent(persistent_false, trials) if not quench else np.nan,
        "simple_median_latency_ms": float(np.median(simple_latency)) if simple_latency else np.nan,
        "persistent_median_latency_ms": float(np.median(persistent_latency)) if persistent_latency else np.nan,
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_basic_checks():
    clean = np.zeros(10)
    clean[4:] = 0.2
    assert first_threshold_crossing(clean, 0.1) == 4
    assert persistent_threshold_crossing(clean, 0.1, 3) == 6
    assert first_threshold_crossing(np.zeros(10), 0.1) is None
    assert persistent_threshold_crossing(np.zeros(10), 0.1, 3) is None
    print("Basic detector checks passed.")


def plot_example_signals():
    rng = np.random.default_rng(SEED)
    ideal_q, measured_q = make_trial(rng, True, NOISE_STD)
    _, measured_spike = make_trial(rng, False, NOISE_STD, spike=True)
    n_validation = max(1, round(VALIDATION_MS / (DT * 1000.0)))
    q_simple, q_persistent = evaluate_detectors(measured_q, THRESHOLD, n_validation)
    s_simple, s_persistent = evaluate_detectors(measured_spike, THRESHOLD, n_validation)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(TIME_MS, ideal_q * 1000, label="Idealised signal", linewidth=2)
    axes[0].plot(TIME_MS, measured_q * 1000, label="Signal + noise", linewidth=0.9)
    axes[0].axhline(THRESHOLD * 1000, linestyle="--", label="Threshold")
    axes[0].axvline(T_QUENCH * 1000, linestyle=":", label="Quench start")
    if q_simple is not None:
        axes[0].axvline(TIME_MS[q_simple], linestyle="-.", label="Simple detector")
    if q_persistent is not None:
        axes[0].axvline(TIME_MS[q_persistent], linestyle="--", label="Validation detector")
    axes[0].set_title("Quench-like signal")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].plot(TIME_MS, measured_spike * 1000, label="Noise + short spike", linewidth=1)
    axes[1].axhline(THRESHOLD * 1000, linestyle="--", label="Threshold")
    if s_simple is not None:
        axes[1].axvline(TIME_MS[s_simple], linestyle="-.", label="Simple trigger")
    if s_persistent is not None:
        axes[1].axvline(TIME_MS[s_persistent], linestyle=":", label="Validation trigger")
    axes[1].set_title("No quench: transient disturbance")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Synthetic signals — not CERN measurements")
    fig.savefig(OUT / "01_signal_examples.png", dpi=160)
    plt.close(fig)


def threshold_sweep():
    thresholds = np.round(np.arange(0.02, 0.201, 0.01), 4)
    n_validation = max(1, round(VALIDATION_MS / (DT * 1000.0)))
    rows = []

    for threshold in thresholds:
        q = trial_summary(True, False, threshold, n_validation,
                          NOISE_STD, N_TRIALS, SEED + 100)
        clean = trial_summary(False, False, threshold, n_validation,
                              NOISE_STD, N_TRIALS, SEED + 500)
        spike = trial_summary(False, True, threshold, n_validation,
                              NOISE_STD, N_TRIALS, SEED + 900)
        rows.append({
            "threshold_mV": round(threshold * 1000.0, 2),
            "simple_post_onset_trigger_pct": q["simple_post_onset_trigger_pct"],
            "persistent_post_onset_trigger_pct": q["persistent_post_onset_trigger_pct"],
            "simple_pre_onset_trigger_pct": q["simple_pre_onset_trigger_pct"],
            "persistent_pre_onset_trigger_pct": q["persistent_pre_onset_trigger_pct"],
            "simple_false_alarm_noise_only_pct": clean["simple_false_alarm_pct"],
            "persistent_false_alarm_noise_only_pct": clean["persistent_false_alarm_pct"],
            "simple_false_alarm_spike_pct": spike["simple_false_alarm_pct"],
            "persistent_false_alarm_spike_pct": spike["persistent_false_alarm_pct"],
            "simple_median_latency_ms": q["simple_median_latency_ms"],
            "persistent_median_latency_ms": q["persistent_median_latency_ms"],
        })
    save_csv(OUT / "threshold_sweep.csv", rows)

    x = [r["threshold_mV"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(x, [r["simple_post_onset_trigger_pct"] for r in rows], marker="o", label="Simple")
    axes[0].plot(x, [r["persistent_post_onset_trigger_pct"] for r in rows], marker="o", label="Validation")
    axes[0].set_title("Triggers after quench onset")
    axes[0].set_xlabel("Threshold (mV)")
    axes[0].set_ylabel("Trials (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["simple_false_alarm_noise_only_pct"] for r in rows], label="Simple, noise only")
    axes[1].plot(x, [r["persistent_false_alarm_noise_only_pct"] for r in rows], label="Validation, noise only")
    axes[1].plot(x, [r["simple_false_alarm_spike_pct"] for r in rows], linestyle="--", label="Simple, with spike")
    axes[1].plot(x, [r["persistent_false_alarm_spike_pct"] for r in rows], linestyle="--", label="Validation, with spike")
    axes[1].set_title("False alarms in no-quench trials")
    axes[1].set_xlabel("Threshold (mV)")
    axes[1].set_ylabel("Trials with a trigger (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Threshold sweep — {N_TRIALS} trials per setting")
    fig.savefig(OUT / "02_threshold_sweep.png", dpi=160)
    plt.close(fig)
    return rows


def validation_sweep():
    validation_ms_values = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    rows = []

    for validation_ms in validation_ms_values:
        n_samples = max(1, round(validation_ms / (DT * 1000.0)))
        q = trial_summary(True, False, THRESHOLD, n_samples,
                          NOISE_STD, N_TRIALS, SEED + 1100)
        clean = trial_summary(False, False, THRESHOLD, n_samples,
                              NOISE_STD, N_TRIALS, SEED + 1500)
        spike = trial_summary(False, True, THRESHOLD, n_samples,
                              NOISE_STD, N_TRIALS, SEED + 1900)
        rows.append({
            "validation_ms": round(n_samples * DT * 1000, 3),
            "persistent_post_onset_trigger_pct": q["persistent_post_onset_trigger_pct"],
            "persistent_pre_onset_trigger_pct": q["persistent_pre_onset_trigger_pct"],
            "persistent_false_alarm_noise_only_pct": clean["persistent_false_alarm_pct"],
            "persistent_false_alarm_spike_pct": spike["persistent_false_alarm_pct"],
            "persistent_median_latency_ms": q["persistent_median_latency_ms"],
        })
    save_csv(OUT / "validation_sweep.csv", rows)

    x = [r["validation_ms"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(x, [r["persistent_post_onset_trigger_pct"] for r in rows], marker="o")
    axes[0].set_title("Triggers after quench onset")
    axes[0].set_xlabel("Validation time (ms)")
    axes[0].set_ylabel("Trials (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, [r["persistent_false_alarm_noise_only_pct"] for r in rows], marker="o", label="Noise only")
    axes[1].plot(x, [r["persistent_false_alarm_spike_pct"] for r in rows], marker="o", label="With spike")
    axes[1].set_title("False alarms in no-quench trials")
    axes[1].set_xlabel("Validation time (ms)")
    axes[1].set_ylabel("Trials with a trigger (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"Validation-time sweep — threshold {THRESHOLD * 1000:.0f} mV")
    fig.savefig(OUT / "03_validation_sweep.png", dpi=160)
    plt.close(fig)
    return rows


def noise_sweep():
    noise_values = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    n_validation = max(1, round(VALIDATION_MS / (DT * 1000.0)))
    rows = []

    for noise_std in noise_values:
        q = trial_summary(True, False, THRESHOLD, n_validation,
                          noise_std, N_TRIALS, SEED + 2100)
        clean = trial_summary(False, False, THRESHOLD, n_validation,
                              noise_std, N_TRIALS, SEED + 2500)
        spike = trial_summary(False, True, THRESHOLD, n_validation,
                              noise_std, N_TRIALS, SEED + 2900)
        rows.append({
            "noise_std_mV": round(noise_std * 1000.0, 3),
            "simple_post_onset_trigger_pct": q["simple_post_onset_trigger_pct"],
            "persistent_post_onset_trigger_pct": q["persistent_post_onset_trigger_pct"],
            "simple_false_alarm_noise_only_pct": clean["simple_false_alarm_pct"],
            "persistent_false_alarm_noise_only_pct": clean["persistent_false_alarm_pct"],
            "simple_false_alarm_spike_pct": spike["simple_false_alarm_pct"],
            "persistent_false_alarm_spike_pct": spike["persistent_false_alarm_pct"],
            "simple_median_latency_ms": q["simple_median_latency_ms"],
            "persistent_median_latency_ms": q["persistent_median_latency_ms"],
        })
    save_csv(OUT / "noise_sweep.csv", rows)

    x = [r["noise_std_mV"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(x, [r["simple_post_onset_trigger_pct"] for r in rows], marker="o", label="Simple")
    axes[0].plot(x, [r["persistent_post_onset_trigger_pct"] for r in rows], marker="o", label="Validation")
    axes[0].set_title("Triggers after quench onset")
    axes[0].set_xlabel("Noise standard deviation (mV)")
    axes[0].set_ylabel("Trials (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["simple_false_alarm_spike_pct"] for r in rows], marker="o", label="Simple, with spike")
    axes[1].plot(x, [r["persistent_false_alarm_spike_pct"] for r in rows], marker="o", label="Validation, with spike")
    axes[1].plot(x, [r["simple_false_alarm_noise_only_pct"] for r in rows], linestyle="--", label="Simple, noise only")
    axes[1].plot(x, [r["persistent_false_alarm_noise_only_pct"] for r in rows], linestyle="--", label="Validation, noise only")
    axes[1].set_title("False alarms in no-quench trials")
    axes[1].set_xlabel("Noise standard deviation (mV)")
    axes[1].set_ylabel("Trials with a trigger (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Noise sweep — threshold {THRESHOLD * 1000:.0f} mV")
    fig.savefig(OUT / "04_noise_sweep.png", dpi=160)
    plt.close(fig)
    return rows


def robustness_heatmap():
    noise_values = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
    threshold_values = np.round(np.arange(0.03, 0.181, 0.01), 4)
    n_validation = max(1, round(VALIDATION_MS / (DT * 1000.0)))
    post_onset = np.zeros((len(noise_values), len(threshold_values)))
    pre_onset = np.zeros_like(post_onset)
    rows = []

    for i, noise_std in enumerate(noise_values):
        for j, threshold in enumerate(threshold_values):
            result = trial_summary(
                True, False, threshold, n_validation,
                noise_std, N_TRIALS, SEED + 3300 + i
            )
            post_onset[i, j] = result["persistent_post_onset_trigger_pct"]
            pre_onset[i, j] = result["persistent_pre_onset_trigger_pct"]
            rows.append({
                "noise_std_mV": round(noise_std * 1000.0, 3),
                "threshold_mV": round(threshold * 1000.0, 2),
                "post_onset_trigger_pct": result["persistent_post_onset_trigger_pct"],
                "pre_onset_trigger_pct": result["persistent_pre_onset_trigger_pct"],
                "validation_ms": round(n_validation * DT * 1000.0, 3),
                "trials": N_TRIALS,
            })
    save_csv(OUT / "robustness_heatmap.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    extent = [threshold_values[0] * 1000, threshold_values[-1] * 1000,
              noise_values[0] * 1000, noise_values[-1] * 1000]
    image1 = axes[0].imshow(post_onset, origin="lower", aspect="auto",
                            extent=extent, vmin=0, vmax=100)
    axes[0].set_title("Trigger after quench onset (%)")
    axes[0].set_xlabel("Threshold (mV)")
    axes[0].set_ylabel("Noise standard deviation (mV)")
    fig.colorbar(image1, ax=axes[0])

    image2 = axes[1].imshow(pre_onset, origin="lower", aspect="auto",
                            extent=extent, vmin=0, vmax=100)
    axes[1].set_title("Trigger before quench onset (%)")
    axes[1].set_xlabel("Threshold (mV)")
    axes[1].set_ylabel("Noise standard deviation (mV)")
    fig.colorbar(image2, ax=axes[1])
    fig.suptitle("Robustness map — validation detector")
    fig.savefig(OUT / "05_robustness_heatmap.png", dpi=160)
    plt.close(fig)
    return rows


def latency_plot(threshold_rows):
    x = [r["threshold_mV"] for r in threshold_rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, [r["simple_median_latency_ms"] for r in threshold_rows], marker="o", label="Simple")
    axes[0].plot(x, [r["persistent_median_latency_ms"] for r in threshold_rows], marker="o", label="Validation")
    axes[0].set_ylabel("Median time after onset (ms)")
    axes[0].set_title("Latency in trials that triggered after quench onset")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["simple_post_onset_trigger_pct"] for r in threshold_rows], marker="o", label="Simple")
    axes[1].plot(x, [r["persistent_post_onset_trigger_pct"] for r in threshold_rows], marker="o", label="Validation")
    axes[1].set_xlabel("Threshold (mV)")
    axes[1].set_ylabel("Post-onset trigger rate (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(OUT / "06_latency_tradeoff.png", dpi=160)
    plt.close(fig)


def main():
    run_basic_checks()
    plot_example_signals()
    thresholds = threshold_sweep()
    validation_sweep()
    noise_sweep()
    robustness_heatmap()
    latency_plot(thresholds)

    n_validation = max(1, round(VALIDATION_MS / (DT * 1000.0)))
    rng = np.random.default_rng(SEED)
    _, measured = make_trial(rng, True, NOISE_STD)
    simple, persistent = evaluate_detectors(measured, THRESHOLD, n_validation)

    lines = [
        "QUENCH-DETECTION SIMULATION",
        "All signals and settings are synthetic and illustrative.",
        f"Number of samples: {TIME.size}",
        f"Time step: {DT * 1000:.3f} ms",
        f"Quench-like onset: {T_QUENCH * 1000:.1f} ms",
        f"Noise standard deviation: {NOISE_STD * 1000:.1f} mV",
        f"Threshold: {THRESHOLD * 1000:.1f} mV",
        f"Validation period: {n_validation * DT * 1000:.1f} ms",
        f"Example simple trigger: {TIME_MS[simple]:.3f} ms" if simple is not None else "Example simple trigger: none",
        f"Example validation trigger: {TIME_MS[persistent]:.3f} ms" if persistent is not None else "Example validation trigger: none",
        "",
        "Plots:",
        "01_signal_examples.png",
        "02_threshold_sweep.png",
        "03_validation_sweep.png",
        "04_noise_sweep.png",
        "05_robustness_heatmap.png",
        "06_latency_tradeoff.png",
        "",
        "CSV results:",
        "threshold_sweep.csv",
        "validation_sweep.csv",
        "noise_sweep.csv",
        "robustness_heatmap.csv",
        "",
        "Limits: no magnet circuit, quench propagation, thermal model, real sensor model, or CERN measurements.",
    ]
    (OUT / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Basic checks passed. Simulation finished.")
    print(f"Results folder: {OUT}")
    print("Open run_summary.txt for the output list.")


if __name__ == "__main__":
    main()
