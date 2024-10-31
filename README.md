# KantoDex-Classifier

A project to identify Generation I Pokémon using deep neural networks.



## Setup Instructions

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/)
- [Kaggle API](https://www.kaggle.com/docs/api)
- [Kaggle Account](https://www.kaggle.com/account)

### 1. Clone the Repository

```bash
git clone https://github.com/arnormoncada/KantoDex-Classifier.git
cd KantoDex-Classifier
```

### 2. Set Up the Environment

```bash
./setup_env.sh
```

### 3. Download the Dataset

Ensure you have set up Kaggle API credentials. You can create a .env file similar to the .env.example file but with yor own credentials.

```bash
python download_dataset.py
```

### 4. Train the Model

```bash
python train.py
```