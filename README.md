# EEG Sleep-Stage Classification

A research-style pipeline for sleep-stage classification using real electroencephalography (EEG) data from the Sleep-EDF Expanded database. The system loads raw polysomnography (PSG) recordings, extracts EEG channels, segments them into 30-second epochs aligned with expert-scored hypnograms, and compares multiple classifiers: 1D CNN, Logistic Regression, and Random Forest.

## Background

### EEG Basics

Electroencephalography (EEG) records electrical activity of the brain via electrodes placed on the scalp. The signal reflects postsynaptic potentials from cortical pyramidal neurons. EEG is central to sleep medicine because brain oscillations change characteristically across sleep stages.

Key frequency bands:

| Band | Frequency | Characteristics |
|------|-----------|-----------------|
| Delta | 0.5–4 Hz | High-amplitude slow waves; dominant in deep sleep (N3) |
| Theta | 4–8 Hz | Present in drowsiness and light sleep (N1) |
| Alpha | 8–13 Hz | Posterior dominant rhythm; prominent during relaxed wakefulness with eyes closed |
| Beta | 13–30 Hz | Low-amplitude fast activity; associated with active waking and REM |

### Sleep Stages

Sleep is classified into five stages per the AASM (American Academy of Sleep Medicine) standard:

- **Wake (W)**: Low-amplitude, mixed-frequency EEG with alpha rhythm when eyes closed; eye blinks and voluntary eye movements present.
- **N1**: Transitional stage; theta activity replaces alpha; vertex sharp waves may appear.
- **N2**: Characterized by sleep spindles (11–16 Hz bursts) and K-complexes; constitutes ~45–55% of total sleep.
- **N3**: Deep or slow-wave sleep; high-amplitude (>75 μV) delta activity; also known as slow-wave sleep (SWS). In the R&K standard (used by Sleep-EDF), this is split into stages 3 and 4.
- **REM**: Rapid eye movements; mixed-frequency, low-amplitude EEG similar to wake; skeletal muscle atonia; dreaming most vivid.

### Sleep-EDF Dataset

The Sleep-EDF Expanded database (PhysioNet) contains overnight PSG recordings from healthy subjects. Each recording includes:

- EEG channels (Fpz-Cz and Pz-Oz), EOG, EMG, and respiratory signals
- Expert-scored hypnograms at 30-second resolution
- Sampling rate of 100 Hz

This project uses the SC4001 recording (subject 1 from the Sleep-EDF Expanded corpus): 22 hours, 7 channels, 2650 valid 30-second epochs (Wake 75%, N1 2%, N2 9%, N3 8%, REM 5%).

## Project Structure

```
├── dataset/
│   ├── SC4001E0-PSG.edf          # Polysomnography recording
│   └── SC4001EC-Hypnogram.edf    # Expert-scored hypnogram
├── figures/                       # Generated visualizations (15+ figures)
├── data_loader.py                 # EDF loading and annotation parsing
├── preprocess.py                  # Epoch segmentation and normalization
├── feature_extraction.py          # Band power and statistical feature extraction
├── visualize.py                   # EEG visualization and spectral analysis
├── model.py                       # 1D CNN architecture
├── train.py                       # Training loop (baseline + class-balanced)
├── evaluate.py                    # Performance metrics and confusion matrix
├── compare_training.py            # Baseline vs. class-balanced training comparison
├── classical_baselines.py         # Logistic Regression and Random Forest baselines
├── quantitative_eeg_analysis.ipynb # Spectral analysis across sleep stages
├── cnn_feature_analysis.ipynb     # CNN feature map interpretability analysis
├── main.py                        # Orchestrates the full pipeline
├── requirements.txt
└── README.md
```

## Pipeline

### 1. Data Loading (`data_loader.py`)

Loads the PSG and hypnogram EDF files using MNE. Automatically detects available channels and selects `EEG Fpz-Cz` (with `EEG Pz-Oz` as fallback). Parses annotations from the hypnogram file which contain variable-duration sleep stage labels. Maps R&K stages (1, 2, 3, 4, R, W) to the 5-class AASM system, merging stages 3 and 4 into N3. Unknown/movement labels are discarded.

### 2. Preprocessing (`preprocess.py`)

Segments the continuous EEG into non-overlapping 30-second epochs (3000 samples at 100 Hz). Each epoch is z-score normalized (zero mean, unit variance) to reduce inter-recording variability. The data is split into train/val/test sets (60/20/20) with stratification to preserve class distribution.

### 3. Feature Extraction (`feature_extraction.py`)

Computes 7 hand-crafted features per epoch for classical ML baselines:

- Mean, Variance, RMS (time-domain statistics)
- Delta, Theta, Alpha, Beta band power (via Welch's PSD, 4-second windows)

### 4. Visualization (`visualize.py`)

Generates diagnostic figures including example epochs, FFT/PSD analysis with delta/theta/alpha/beta bands, and average EEG waveforms per sleep stage.

### 5. Quantitative EEG Analysis (`quantitative_eeg_analysis.ipynb`)

A standalone Jupyter notebook computing Welch PSD and band power per sleep stage. Generates PSD overlay curves and bar charts for absolute and relative band power. Includes physiological interpretation: N3 delta dominance, Wake alpha/beta activity, and spectral similarity between N1 and REM.

### 6. CNN Model (`model.py`)

A compact 1D convolutional neural network designed for CPU training:

| Layer | Configuration | Output |
|-------|--------------|--------|
| Conv1D | 16 filters, kernel 7, BN, ReLU | 16 × 3000 |
| MaxPool | Kernel 2 | 16 × 1500 |
| Conv1D | 32 filters, kernel 5, BN, ReLU | 32 × 1500 |
| MaxPool | Kernel 2 | 32 × 750 |
| Conv1D | 64 filters, kernel 3, BN, ReLU | 64 × 750 |
| MaxPool | Kernel 2 | 64 × 375 |
| Flatten | — | 24000 |
| Dropout | p = 0.3 | 24000 |
| Dense | 64 units, ReLU | 64 |
| Dense | 5 units | 5 |

Total parameters: ~1.5M. Batch normalization after each convolution for stability; dropout for regularization.

### 7. Training (`train.py`)

Supports two modes via the `balanced` flag:

- **Baseline**: Standard CrossEntropyLoss, random batch shuffling
- **Class-balanced**: Class-weighted CrossEntropyLoss + WeightedRandomSampler to oversample minority stages

Training: Adam (lr=0.001), batch size 32, 10 epochs, StepLR scheduler (×0.5 every 5 epochs).

### 8. CNN Feature Analysis (`cnn_feature_analysis.ipynb`)

Extracts intermediate feature maps via forward hooks to visualize:

- Conv1 learned filters (16 temporal kernels, 70ms each)
- Activation maps across Wake/N2/N3/REM
- Filter selectivity (most discriminative filter per stage)
- Deep layer (conv2/conv3) activation distributions
- Frequency response of learned filters via DFT
- Raw EEG waveform vs. filter response comparison

### 9. Classical ML Baselines (`classical_baselines.py`)

Trains Logistic Regression and Random Forest on the 7 hand-crafted features and compares against the CNN. Generates a side-by-side comparison table and discussion of feature-based vs. deep learning approaches.

## Results

### CNN Baseline Performance

| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Wake | 0.99 | 0.98 | 0.99 | 399 |
| N1 | 0.29 | 0.33 | 0.31 | 12 |
| N2 | 0.77 | 0.88 | 0.82 | 50 |
| N3 | 0.89 | 0.91 | 0.90 | 44 |
| REM | 0.60 | 0.48 | 0.53 | 25 |

**Accuracy: 92.8%  |  Macro F1: 0.710**

### Class-Balanced Training Impact

Balanced training uses class-weighted loss + WeightedRandomSampler:

| Class | Baseline F1 | Balanced F1 | Change |
|-------|:-----------:|:-----------:|:------:|
| Wake | 0.965 | 0.912 | −0.05 |
| **N1** | **0.000** | **0.158** | **+0.16** |
| N2 | 0.779 | 0.741 | −0.04 |
| N3 | 0.805 | 0.854 | +0.05 |
| **REM** | **0.133** | **0.317** | **+0.18** |
| **Macro F1** | **0.536** | **0.596** | **+0.06** |

N1 went from completely unpredicted to detecting 25% of true N1s. REM recall jumped from 8% → 52%. The trade-off is a modest dip in Wake/N2 F1.

### Classical ML Baselines vs. CNN

| Model | Accuracy | Macro F1 | Wake | N1 | N2 | N3 | REM |
|-------|:--------:|:--------:|:----:|:--:|:--:|:--:|:---:|
| Logistic Regression | 0.772 | 0.269 | 0.870 | 0.000 | 0.000 | 0.476 | 0.000 |
| **Random Forest** | **0.943** | **0.772** | 0.985 | 0.375 | 0.807 | **0.929** | **0.766** |
| CNN | 0.928 | 0.710 | **0.989** | 0.308 | **0.822** | 0.899 | 0.533 |

**Key finding:** Random Forest (with `class_weight="balanced"`) outperforms the 1.5M-parameter CNN on macro F1 (+0.06) and per-class F1 for N3 (+0.03) and especially REM (+0.23). CNN only edges ahead on Wake and N2. This suggests that for single-channel, single-subject sleep staging, hand-crafted spectral features capture most of the relevant discriminative information — the temporal structure learned by the CNN adds marginal value for this particular task.

### Feature Importance (Random Forest)

The RF feature importance ranking: Delta > Beta > Variance > Theta > Alpha > RMS > Mean

Delta power is the single most important feature, consistent with the physiological fact that delta activity is the primary discriminator between sleep stages (especially N3 vs. all others).

## Interpretability Analysis

### Filter Frequency Response

All 16 conv1 filters have center frequencies in the 18–31 Hz range (beta band). This is because the kernel size (7 samples = 70ms at 100 Hz) is too short to resolve a full cycle of delta (0.5–4 Hz, 250–2000ms). The CNN builds low-frequency selectivity combinatorially across deeper layers rather than through individual first-layer filters.

### What the CNN Learns

- **Layer 1**: Edge detectors and oscillatory templates (70ms windows) — primarily high-frequency transient detectors
- **Layer 2**: Combinations of conv1 patterns across longer time spans (1500 samples after pooling)
- **Layer 3**: Whole-epoch macro-patterns (375-length feature vectors before the classifier)

## Usage

```bash
python main.py
```

Outputs:
- `figures/example_epochs.png`, `figures/fft_psd_analysis.png`, `figures/stage_averages.png`
- `figures/confusion_matrix.png`, `figures/training_comparison.png`
- `figures/classical_vs_cnn.png`, `figures/psd_by_stage.png`, `figures/band_power_bars.png`
- `figures/conv1_filters.png`, `figures/conv1_activations.png`, `figures/filter_selectivity.png`
- `figures/conv2_activations.png`, `figures/conv3_activations.png`
- `figures/eeg_vs_filter_response.png`, `figures/filter_frequency_response.png`
- `sleep_stage_cnn.pth`, `sleep_stage_cnn_baseline.pth`, `sleep_stage_cnn_balanced.pth`

Individual analysis scripts can be run standalone:

```bash
python compare_training.py        # Baseline vs. balanced comparison
python classical_baselines.py     # Classical ML baselines
python quantitative_eeg_analysis.ipynb  # Spectral analysis (Jupyter)
python cnn_feature_analysis.ipynb       # CNN interpretability (Jupyter)
```

## Requirements

- Python ≥ 3.8
- MNE
- NumPy
- PyTorch
- scikit-learn
- Matplotlib
- SciPy
- Jupyter (for notebooks)

## Limitations

- Single subject recording — inter-subject variability not addressed.
- Single EEG channel (Fpz-Cz) — clinical scoring uses multiple channels.
- Class imbalance: Wake dominates (75%), N1 and REM are underrepresented.
- The dataset uses the older R&K standard (stages 3 and 4 separate); merged into AASM N3.
- No artifact rejection or advanced preprocessing (ICA, bandpass filtering).
- CNN kernel size (7) limits low-frequency resolution in the first layer — deeper layers must compensate.
- Classical ML features (band powers) may miss transient events like spindles and K-complexes.

## References

- Kemp, B., et al. (2000). "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG." *IEEE Transactions on Biomedical Engineering*, 47(9), 1185–1194.
- Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." *Circulation*, 101(23), e215–e220.
- Berry, R. B., et al. (2012). "The AASM Manual for the Scoring of Sleep and Associated Events." *American Academy of Sleep Medicine*.
