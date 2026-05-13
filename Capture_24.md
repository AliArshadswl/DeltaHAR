# Dataset & Evaluation Notes
# (Give this file to an LLM at the start of any new session)

---

## Dataset

The dataset is located at /mnt/share/ali/processed/

It contains 151 subjects. Each subject has one .npz file named P001.npz
through P151.npz. Load each file with:

    d = np.load(path, allow_pickle=True)

Each file contains three arrays:

X   — sensor windows, shape (N, 3000, 3), dtype float32
        N        = number of windows (varies per subject)
        3000     = samples per window (30 seconds at 100 Hz)
        3        = accelerometer channels (X, Y, Z axes)
        windows were created with 50% overlap (stride = 1500 samples)

y   — activity labels, shape (N,), dtype string
        values are activity name strings, e.g. "sleep", "walk", "sit", etc.
        they are NOT pre-encoded as integers
        to get integer indices, scan all 151 files to collect the full set of
        unique label strings, sort them alphabetically, and build:
            class_to_idx = { label_string: integer_index }
        then convert: label_int = class_to_idx[str(label)]

t   — window timestamps, shape (N,), dtype datetime64[ns] stored as int64
        represents the start time of each window
        use this to sort windows chronologically within each subject before
        training, as the order matters for HMM smoothing and temporal splits

---

## Data Loading Rules

- Always use allow_pickle=True when calling np.load
- Always sort windows by t before any processing
- If X arrives as (N, 3, 3000) i.e. channels-first, transpose to (N, 3000, 3)
- Normalise per-window per-channel: (x - mean) / (std + 1e-8)
- Clip normalised values to [-10, 10]
- Never use training set statistics to normalise test windows (data leakage)
- Use instance normalisation (per-window) for a fully leak-free pipeline

---

## Evaluation Protocol

- 5-fold subject-wise cross-validation
- Splits happen at the subject level: no subject's windows appear in both
  train and test within the same fold
- Split ratio: 70% train / 10% val / 20% test by number of subjects
- Shuffle subjects once with a fixed random seed (seed=42) and keep fixed
- Within each fold: train the model, monitor val macro-F1, save the best
  checkpoint, evaluate the best checkpoint on the test set
- Early stopping: stop training if val macro-F1 does not improve for 15 epochs
- After test inference: apply HMM smoothing per subject (see below)
- TTA: average softmax probabilities over the original input and its
  time-reversed version

HMM smoothing:
- Fit a Categorical HMM on the predicted probability streams of all test subjects
- Decode each subject's stream independently using Viterbi
- Report both raw and HMM-smoothed metrics

---

## Evaluation Metrics

Report all of the following, as mean ± std across the 5 folds,
and also pooled (all fold test sets concatenated):

- Macro-F1        primary metric, unweighted mean F1 across all classes
- Weighted-F1     F1 weighted by class support
- Cohen's kappa   agreement above chance
- MCC             Matthews Correlation Coefficient, robust to class imbalance
- HMM Macro-F1   macro-F1 computed on HMM-smoothed predictions
- ECE             Expected Calibration Error, measures confidence calibration

---

## Output Figures

Save all figures to /mnt/share/ali/processed/harmamba_results/

fig00_architecture.png
    architecture diagram or text summary of the model being evaluated

fold{N}_best.pt
    best model checkpoint (state dict) for each of the 5 folds

fig_confusion_matrix_pooled.png
    confusion matrix over all fold test sets combined, raw predictions

fig_confusion_matrix_hmm_pooled.png
    same confusion matrix after HMM temporal smoothing

fig_per_class_f1_pooled.png
    bar chart of F1 per activity class, raw predictions, pooled test set

fig_per_class_f1_hmm_pooled.png
    same per-class F1 bar chart after HMM smoothing

fig_per_participant_f1_pooled.png
    bar chart of macro-F1 per subject, HMM predictions, pooled across folds
    include a red dashed line for the mean across participants

fig_reliability_pooled.png
    reliability diagram: model confidence vs actual accuracy
    binned bar chart with ECE value shown in the legend

fig_umap_embeddings_pooled.png
    2D UMAP projection of the model's output embeddings (e.g. CLS token)
    coloured by ground-truth activity class, pooled across all fold test sets

cv_summary.png
    bar chart of mean ± std for all key metrics across the 5 folds

fig_compute_summary.png
    summary table of model config, training hyperparameters, and CV results

results_table.tex
    LaTeX table of mean ± std results, ready to paste into a paper

compute_profile.json
    all results and config as a machine-readable JSON file

fold_assignments.json
    exact train/val/test subject ID lists for every fold, for reproducibility
