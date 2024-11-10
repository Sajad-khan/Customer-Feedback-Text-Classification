import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data('content/data/customer_reviews.csv')

# Define and train the model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Filter classes in label encoder based on y_test values
y_test_classes = sorted(set(y_test))
target_names = [label_encoder.inverse_transform([cls])[0] for cls in y_test_classes]

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Save the model and label encoder
joblib.dump(pipeline, 'content/model/customer_feedback_classifier.pkl')
joblib.dump(label_encoder, 'content/model/label_encoder.pkl')
