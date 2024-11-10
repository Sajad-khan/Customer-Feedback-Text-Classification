This project classifies customer feedback into categories using Natural Language Processing (NLP) and Machine Learning. By automatically labeling feedback, it enables quick analysis for better insights.

**Overview**
The project reads customer feedback from a CSV, preprocesses text data, trains a classification model, evaluates performance, and saves the model for future predictions.

**Structure**
content/
data_preprocessing.py: Loads and cleans data
model_training.py: Trains and evaluates the model
data/ (optional): Contains customer_reviews.csv if using local data

**Dependencies**
Install dependencies with:
pip install -r requirements.txt

**Usage**
Data Preprocessing: Run data_preprocessing.py to load and clean data.
Model Training: Run model_training.py to train and evaluate the model.
Pipeline: Use main.py (if available) to execute the full process.
The trained model and label encoder are saved in content/model/ for future predictions.
