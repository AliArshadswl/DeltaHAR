# Dataset Notes

## Location
/mnt/share/ali/processed/

## Files
- 151 subjects total
- One file per subject: P001.npz, P002.npz, ... P151.npz
- Each file contains two arrays with keys "X" and "y"

## Data (X)
- Shape: (N, 3000, 3)
  - N = number of windows (varies per subject)
  - 3000 = samples per window
  - 3 = sensor channels (accelerometer, likely X/Y/Z axes)
- dtype: float32
- Sampling rate: 100 Hz
- Window length: 30 seconds (3000 samples @ 100 Hz)
- Overlap: 50% (stride = 1500 samples)

## Labels (y)
- Shape: (N,)
- dtype: string (activity names, e.g. "sleep", "walk", ...)
- NOT integer-encoded — must be mapped to integers at load time
- Full class list is discovered by scanning all 151 subject files
- Mapping: class_to_idx = { class_name: integer_index } (sorted alphabetically)

## Timestamps (t)  [referenced in older code, may exist in some files]
- Key: "t"
- dtype: datetime64[ns] cast to int64
- Used for sorting windows chronologically per subject

## Loading notes
- np.load(path, allow_pickle=True) — allow_pickle must be True
- After loading y, convert each label: class_to_idx[str(label)]
- X may come in (N, C, T) format in some files — transpose to (N, T, C) if so
- Clip normalised values to [-10, 10] after per-channel z-score normalisation

## Normalisation
- Per-window, per-channel z-score: (x - mean) / (std + 1e-8)
- Applied at load time inside the Dataset class
- Training set stats must NOT be used on test windows (leakage risk)
- Recommended: instance normalisation (per-window) for leak-free evaluation

---

## Evaluation Protocol
- 5-fold subject-wise cross-validation
- Splits are done at subject level — no subject appears in both train and test
- Ratio: 70% train / 10% val / 20% test (by number of subjects)
- Fold assignments shuffled once with a fixed seed (seed=42) and held constant
- Early stopping: patience = 15 epochs, monitored on validation macro-F1
- Best checkpoint = epoch with highest val macro-F1
- Post-hoc HMM smoothing: Categorical HMM fitted on per-subject probability streams, decoded with Viterbi
- TTA: softmax averaged over original input and time-reversed input

## Evaluation Metrics
- Macro-F1        — primary metric (unweighted mean F1 across classes)
- Weighted-F1     — F1 weighted by class support
- Cohen's kappa   — agreement above chance
- MCC             — Matthews Correlation Coefficient (robust to imbalance)
- HMM Macro-F1   — macro-F1 after HMM temporal smoothing
- ECE             — Expected Calibration Error (confidence vs accuracy)
- All reported as mean ± std across 5 folds
- Also reported pooled: all fold test sets concatenated into one set
