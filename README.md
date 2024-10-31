# 🚀 KantoDex-Classifier Project

## 🌟 Overview

Welcome to the **KantoDex-Classifier** project! This repository is designed to identify Generation I Pokémon using deep neural networks. 

## 📁 Project Structure

```
KantoDex-Classifier/
├── .env.example
├── .gitignore
├── .ruff.toml
├── README.md
├── download_dataset.py
├── environment.yml
├── setup_env.sh
└── src/
    ├── __init__.py
    ├── augmentation/
    │   ├── __init__.py
    │   └── augmentor.py
    ├── config/
    │   └── config.yaml
    ├── data/
    │   ├── __init__.py
    │   └── data_loader.py
    ├── models/
    │   ├── __init__.py
    │   ├── custom_model.py
    │   └── model.py
    └── utils/
        ├── __init__.py
        └── helpers.py
```

## 🛠️ Included Components

### 1. **Source Code (`src/`):**
   - **Models (`src/models/`):**
     - `model.py`: Defines the `KantoDexClassifier`, a custom neural network tailored for Pokémon classification.
     - `custom_model.py`: Implements advanced architectural components like positional encoding, SE blocks, and multi-head self-attention.
   - **Data Loading (`src/data/`):**
     - `data_loader.py`: Contains the `PokemonDataset` class and functions to load and preprocess the dataset.
   - **Augmentation (`src/augmentation/`):**
     - `augmentor.py`: Implements data augmentation techniques to enhance model robustness.
   - **Utilities (`src/utils/`):**
     - `helpers.py`: Provides helper functions for checkpointing and directory management.
   - **Configuration (`src/config/`):**
     - `config.yaml`: YAML configuration file outlining parameters for data processing, training, augmentation, and model settings.

### 2. **Scripts:**
   - `setup_env.sh`: Shell script to set up the Conda environment based on `environment.yml`.
   - `download_dataset.py`: Script to download and organize the Pokémon dataset from Kaggle.
   - `train.py`: Training script for the model, integrated with TensorBoard for real-time monitoring.

### 3. **Configuration Files:**
   - `.env.example`: Example environment variables file for Kaggle API credentials.
   - `environment.yml`: Conda environment specification listing all dependencies.
   - `.ruff.toml`: Configuration for Ruff, the linter, ensuring code quality and consistency.
   - `.gitignore`: Specifies files and directories to be ignored by Git to maintain repository cleanliness.


## 📋 Setup Instructions

### 1. **Clone the Repository**
```bash
git clone https://github.com/arnormoncada/KantoDex-Classifier.git
cd KantoDex-Classifier
```

### 2. **Set Up the Environment**
```bash
./setup_env.sh
```

### 3. **Configure Kaggle API**
- Rename `.env.example` to `.env` and populate it with your Kaggle API credentials.
  
  **Example:**
  ```bash
  cp .env.example .env
  nano .env
  ```
  
  **.env:**
  ```
  # Kaggle API credentials
  KAGGLE_USERNAME=your_kaggle_username
  KAGGLE_KEY=your_kaggle_key
  ```

### 4. **Download the Dataset**
Ensure you have set up Kaggle API credentials. Execute the script to download and organize the dataset.

```bash
python download_dataset.py
```

### 5. **Train the Model**
```bash
python train.py
```

#### 5.1 **Resume Training**
If you wish to continue training from the latest checkpoint:

```bash
python train.py --resume
```

## 📊 Using TensorBoard

TensorBoard is integrated into the training pipeline to monitor metrics like loss and accuracy in real-time.

1. **Launch TensorBoard**
   ```bash
   tensorboard --logdir=runs
   ```
2. **Access TensorBoard**
   - Open your browser and navigate to [http://localhost:6006/](http://localhost:6006/) to view the dashboard.

## 📝 Additional Notes

- **Environment Consistency:** Ensure that you are using Python 3.11 as specified in the `environment.yml`.
- **Dependency Management:** All dependencies are managed via Conda. If you encounter issues, verify that the environment is correctly set up.
- **Code Quality:** The project adheres to PEP 8 standards, enforced by Ruff. Ensure that all new code passes linting checks.
- **Data Security:** The `.env` file containing Kaggle API credentials is excluded from version control to maintain security.