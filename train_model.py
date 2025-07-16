"""
AI-Based Phishing Detection Model Trainer
-----------------------------------------

This script trains a machine learning model to detect phishing messages based on text input,
such as emails, SMS, or URLs. It uses a Random Forest Classifier and TF-IDF vectorization
to build a predictive model from labeled training data.

Output:
- model.pkl         : Trained Random Forest model
- vectorizer.pkl    : Fitted TF-IDF vectorizer

Note: This demo uses a small in-memory dataset. For production, use a large, diverse dataset.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ------------------------------
# Step 1: Define a small sample dataset
# ------------------------------
# Each row is a sample with:
# - `text`: the email or URL content
# - `label`: 1 for phishing, 0 for legitimate
data = {
    'text': [
        "Click here to verify your bank account",
        "Welcome to your new inbox",
        "Update your password to secure your account",
        "Team lunch at 12pm today",
        "Urgent: Your PayPal has been restricted",
        "Your invoice is ready for download",
        "Reset your account password now",
        "Looking forward to the meeting tomorrow"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# ------------------------------
# Step 2: Feature extraction using TF-IDF
# ------------------------------
# Convert the text input into a numerical representation
vectorizer = TfidfVectorizer(stop_words='english')  # Remove common stopwords
X = vectorizer.fit_transform(df['text'])  # Sparse matrix of text features
y = df['label']  # Target labels

# ------------------------------
# Step 3: Train-test split
# ------------------------------
# Split the data into training and testing sets
# Use stratify=y to preserve label distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------
# Step 4: Train a Random Forest model
# ------------------------------
# A robust, non-linear classifier suitable for tabular/text data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Evaluate the model
# ------------------------------
# Show performance metrics such as precision, recall, F1-score
report = classification_report(y_test, model.predict(X_test))
print("Model Evaluation Report:\n")
print(report)

# ------------------------------
# Step 6: Save model and vectorizer
# ------------------------------
# Save the trained model and vectorizer for use in the inference engine
joblib.dump(model, 'model.pkl')            # Serialized model
joblib.dump(vectorizer, 'vectorizer.pkl')  # Serialized TF-IDF transformer
