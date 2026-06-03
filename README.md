# EEG Sleep-Stage Classification

A  pipeline for sleep-stage classification using real electroencephalography (EEG) data from the Sleep-EDF Expanded database. The system loads raw polysomnography (PSG) recordings, extracts EEG channels, segments them into 30-second epochs aligned with expert-scored hypnograms, and trains a 1D convolutional neural network (CNN) for automatic sleep staging.

## Background

### EEG Basics

Electroencephalography (EEG) records electrical activity of the brain via electrodes placed on the scalp. The signal reflects postsynaptic potentials from cortical pyramidal neurons. EEG is central to sleep medicine because brain oscillations change characteristically across sleep stages.

Key frequency bands:

| Band | Frequency | Characteristics |
|------|-----------|-----------------|
| Delta | 0.5–4 Hz | High-amplitude slow waves; dominant in deep sleep (N3) |
| Theta | 4–8 Hz | Present in drowsiness and light sleep (N1) |
| Alpha | 8–12 Hz | Posterior dominant rhythm; prominent during relaxed wakefulness with eyes closed |
| Beta | 12–30 Hz | Low-amplitude fast activity; associated with active waking and REM |

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

This project uses the SC4001 recording (subject 1 from the Sleep-EDF Expanded corpus).

## Project Structure

```
├── dataset/
│   ├── SC4001E0-PSG.edf          # Polysomnography recording
│   └── SC4001EC-Hypnogram.edf    # Expert-scored hypnogram
├── figures/                       # Generated visualizations
├── data_loader.py                 # EDF loading and annotation parsing
├── preprocess.py                  # Epoch segmentation and normalization
├── visualize.py                   # EEG visualization and spectral analysis
├── model.py                       # 1D CNN architecture
├── train.py                       # Training loop
├── evaluate.py                    # Performance metrics and confusion matrix
├── main.py                        # Orchestrates the full pipeline
├── sleep_stage_cnn.pth            # Trained model weights
└── requirements.txt
```

## Pipeline

### 1. Data Loading (`data_loader.py`)

Loads the PSG and hypnogram EDF files using MNE. Automatically detects available channels and selects `EEG Fpz-Cz` (with `EEG Pz-Oz` as fallback). Parses annotations from the hypnogram file which contain variable-duration sleep stage labels. Maps R&K stages (1, 2, 3, 4, R, W) to the 5-class AASM system, merging stages 3 and 4 into N3. Unknown/movement labels are discarded.

### 2. Preprocessing (`preprocess.py`)

Segments the continuous EEG into non-overlapping 30-second epochs (3000 samples at 100 Hz). Each epoch is z-score normalized (zero mean, unit variance) to reduce inter-subject and inter-recording variability. The data is split into train/val/test sets (60/20/20) with stratification to preserve class distribution.

### 3. Visualization (`visualize.py`)

Generates three diagnostic figures:

- **Example epochs**: Representative 30-second EEG traces for Wake, N2, N3, and REM stages.
- **FFT and PSD analysis**: Frequency-domain comparison across stages using FFT magnitude and Welch's PSD estimate, with delta/theta/alpha/beta bands highlighted.
- **Stage averages**: Mean EEG waveform per sleep stage, revealing stage-specific morphologies.

### 4. Model (`model.py`)

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
| Dense | 5 units, Softmax | 5 |

Total parameters: ~1.5M. The architecture uses batch normalization after each convolution for training stability and dropout for regularization.

### 5. Training (`train.py`)

- Loss: Cross-entropy
- Optimizer: Adam (lr = 0.001)
- Batch size: 32
- Epochs: 10
- Learning rate schedule: StepLR (reduce by 0.5 every 5 epochs)
- Validation accuracy tracked after each epoch

### 6. Evaluation (`evaluate.py`)

Reports overall accuracy, per-class precision/recall/F1, and generates a confusion matrix.

## Results

| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Wake | 0.99 | 0.98 | 0.99 | 399 |
| N1 | 0.29 | 0.33 | 0.31 | 12 |
| N2 | 0.77 | 0.88 | 0.82 | 50 |
| N3 | 0.89 | 0.91 | 0.90 | 44 |
| REM | 0.60 | 0.48 | 0.53 | 25 |

**Overall accuracy: 92.8%**

### Interpretation

The high overall accuracy is driven primarily by Wake classification (75% of epochs). The model performs well on Wake, N2, and N3 but struggles with N1 and REM:

- **N1** is poorly classified (F1 = 0.31) because it is a brief transitional stage with only 12 test samples and its EEG pattern resembles both wake and N2.
- **REM** has modest performance (F1 = 0.53) partly due to class imbalance (4.7% of epochs) and the difficulty of distinguishing REM from wake based on a single EEG channel without EOG context.

These limitations are well-documented in the sleep-staging literature and highlight the challenge of single-channel EEG classification.

## Requirements

- Python ≥ 3.8
- MNE
- NumPy
- PyTorch
- scikit-learn
- Matplotlib
- SciPy

## Usage

```bash
python main.py
```

Outputs:
- `figures/example_epochs.png`
- `figures/fft_psd_analysis.png`
- `figures/stage_averages.png`
- `figures/confusion_matrix.png`
- `sleep_stage_cnn.pth`

## Limitations

- Single subject recording — inter-subject variability not addressed.
- Single EEG channel (Fpz-Cz) — clinical scoring uses multiple channels.
- Class imbalance: Wake dominates (75%), N1 and REM are underrepresented.
- The dataset uses the older R&K standard (stages 3 and 4 separate); merged into AASM N3.
- No artifact rejection or advanced preprocessing (ICA, bandpass filtering).

## References

- Kemp, B., et al. (2000). "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG." *IEEE Transactions on Biomedical Engineering*, 47(9), 1185–1194.
- Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." *Circulation*, 101(23), e215–e220.
- Berry, R. B., et al. (2012). "The AASM Manual for the Scoring of Sleep and Associated Events." *American Academy of Sleep Medicine*.
