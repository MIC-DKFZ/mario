# MARIO : Monitoring AMD progression in OCT (MIC Group 6)
Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.

This repository contains the implementation of the 2nd place method from the 2024 MICCAI MARIO Challenge.

## Getting Started
Follow these steps to replicate our results:

### 1. Setup and Install Dependencies
First, clone the repository and set up the necessary environment:
```bash
git clone [repository URL]
cd mario
conda env create -f environment.yml
```

### 2. Preprocess the Dataset
Our model requires the dataset to be in `.npy` format for efficiency. Please convert the files from `.png` to `.npy` before proceeding.

### 3. Setup Environment Variables
Configure the following environment variables according to your setup:
- `SAVE_DIR_RESULTS`: Directory where models, evaluations, and other outputs will be saved.
- `DATA_DIR`: Path to the MARIO dataset.

### 4. Select Correct Splits File
In `main.py`, line 175, insert the correct splits file:
- Use `splits.json` for a split on the Training set.
- Use `splits_train_all.json` for a split on the Training + Validation set.

### 5. Train the Model
#### Siamese Network
To train the Siamese Network, execute:
```bash
python main.py
```

#### AutoEncoder
For the AutoEncoder, follow these steps:
1. Run the following script to catalog all available images. Ensure the output file is saved in the directory where you intend to store your model results:
   ```bash
   python utils.find_images.py
   ```
2. Start the training process by running:
   ```bash
   python main_ae.py
   ```
  
## Paper
If you use this code in you research, please cite the following paper (TODO)