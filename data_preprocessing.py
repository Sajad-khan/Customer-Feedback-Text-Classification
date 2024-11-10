import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath, encoding='latin-1')
    df.dropna(subset=['Review Title', 'Comments'], inplace=True)

    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    # Apply cleaning
    df['cleaned_text'] = df['Review Title'].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    le.fit(df['Comments'].unique())
    df['label'] = le.transform(df['Comments'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, le

