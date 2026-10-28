# Quench-detection simulation

A small Python benchmark for comparing two voltage-threshold detectors on synthetic signals.

## Run it

Install the two libraries if needed:

```powershell
py -m pip install numpy matplotlib
```

Run the script from this folder:

```powershell
py .\quench_detection_simulator.py
```

It runs a few basic checks, then creates a `quench_results` folder containing graphs, CSV files and a run summary. Running the script again replaces those result files.

## What it simulates

- A voltage that starts at zero and rises linearly at a chosen time.
- Gaussian measurement noise.
- A no-quench case with noise only.
- A no-quench case with a short voltage spike.

The detector comparison is simple:

1. **Threshold detector:** triggers at the first sample at or above the threshold.
2. **Validation detector:** triggers only after the signal stays at or above the threshold for a set number of consecutive samples.

The validation period is converted to a number of samples using the simulation time step.

## Outputs

- `01_signal_examples.png` — example quench-like waveform and transient spike.
- `02_threshold_sweep.png` — vary the detection threshold.
- `03_validation_sweep.png` — vary the validation period.
- `04_noise_sweep.png` — vary the noise level.
- `05_robustness_heatmap.png` — post-onset and pre-onset trigger rates across threshold/noise combinations.
- `06_latency_tradeoff.png` — median trigger latency and post-onset trigger rate versus threshold.
- `threshold_sweep.csv`, `validation_sweep.csv`, `noise_sweep.csv`, `robustness_heatmap.csv` — tabular results.
- `run_summary.txt` — run settings and output list.

## How the metrics are defined

- **Post-onset trigger:** the detector triggers at or after the chosen synthetic quench-start time.
- **Pre-onset trigger:** the detector triggers before that time during a simulated quench trial.
- **False alarm:** the detector triggers in a case where no quench was inserted.
- **Latency:** trigger time minus the synthetic quench-start time, calculated only for post-onset triggers.

A post-onset trigger is not proof that a real quench was identified correctly. The model is deliberately simple, so these metrics describe behaviour under these assumptions only.

## Assumptions and limits

All waveforms and parameter values are synthetic. The script does not model superconducting physics, inductive-voltage compensation, quench propagation, magnet thermal behaviour, energy extraction, actual sensors, or CERN hardware. It does not use measured CERN data and has not been validated against an experiment. The voltage threshold, noise levels, rise rate and transient are illustrative, not settings recommended for a real magnet.

The random seed is fixed so runs are repeatable. The sweeps use the same seeded noise patterns at different parameter settings to make comparisons less noisy; results are still limited by the chosen number of trials. The default is 150 trials per setting. Increase `N_TRIALS` for more stable estimates after checking runtime.

## Background reading

These sources describe real quench-detection and protection work; they are context, not validation data for this simulation.

- [Safeguarding the superconducting magnets — CERN Courier](https://cern-courier.web.cern.ch/a/safeguarding-the-superconducting-magnets/)
- [HFM Quench Detection and Protection — Summary](https://indico.cern.ch/event/1012691/contributions/4291241/attachments/2226583/3800320/HFM%20quench%20detection%20and%20protection%20-%20summary.pdf)
- [The Protection System for the Superconducting Elements of the LHC at CERN — CERN Document Server](https://cds.cern.ch/record/386677)
