# PII Data Detection

This repository contains the source code and resources for an NER project aimed at detecting PII Data. The project is based on the [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) and utilizes a fine-tuned DeBERTa V3 base model to achieve its goal.

## Project Structure

- **`data/`**: Contains training and test datasets in json format. The training and test set provided are placeholders and should be replaced for actual use.
- **`model_checkpoints/`**: Stores trained models' checkpoints.
- **`EDA.ipynb`**: Jupyter notebook for exploratory data analysis on the training set.
- **`model_training.py`**: Fine-tunes the pre-trained DeBERTa V3 base model on the training set and saves the checkpoint to the `model_checkpoints/` folder.
- **`inference.py`**: Loads fine-tuned DeBERTa model to make predictions on the test set.


## Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/q-xZzz/pii-data-detection.git

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt

## Running the Project

To get the project up and running, follow these steps:
- **(optional)Exploratory Data Analysis**: Open `EDA.ipynb` with Jupyter Notebook or JupyterLab to explore the training dataset.
- **Model Training**:
  - Run `python model_training.py`.
- **Inference**: Run `python inference.py` to load the model checkpoints and use it to make predictions on the test set.
