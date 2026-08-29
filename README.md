# SIH (Smart India Hackathon) Project

This repository contains the source code and pre-trained machine learning models for the SIH project.

## Repository Contents

- `cloud_assistant.py`: Script for handling the cloud-based assistant features.
- `cpu.py`: Script optimized for local/CPU execution.
- `train.py`: Script for training models.
- `test.py`: Script for running tests and validation.
- `requirements.txt`: List of Python library dependencies required for this project.
- `*.keras`, `*.tflite`, `*.cc`: Pre-trained neural network models (float and quantized INT8 models for microcontrollers/CPUs).

---

## Ignored Folders and Files

To keep the repository clean, lightweight, and fast to download, the following folders are excluded from Git (as defined in `.gitignore`):

1. **`sih_env/` & `sih_e/` (Virtual Environments)**
   - **Size**: ~2.06 GB
   - **Reason for ignoring**: Contains thousands of local Python binaries and libraries which are platform-specific. These should not be checked into Git. You can recreate the environment using `requirements.txt`.

2. **`vikram_kws_dataset/` & `vikram_split_dataset/` (Audio Datasets)**
   - **Size**: ~530 MB
   - **Reason for ignoring**: Large binary audio files. It is best practice to store datasets externally (e.g., Google Drive, Kaggle, or S3) rather than tracking them in source control.
   - **Usage**: To train the models locally, obtain the audio datasets and place them in the root directory under these folder names.

---

## Setup Instructions

### 1. Recreate the Python Environment

To install all the exact libraries and dependencies used in this project:

1. Create a new virtual environment:
   ```bash
   python -m venv sih_env
   ```
2. Activate the virtual environment:
   - **Windows (Command Prompt)**:
     ```cmd
     sih_env\Scripts\activate.bat
     ```
   - **Windows (PowerShell)**:
     ```powershell
     sih_env\Scripts\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     source sih_env/bin/activate
     ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Dataset Setup
If you plan to run `train.py` or modify the model training, download the `vikram_kws_dataset` and `vikram_split_dataset` folders and place them in the root directory.
