# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

# Load dataset
df = pd.read_csv("assignment3_II.csv")
df = df[['Review Text', 'Rating', 'Recommended IND']].dropna()

# Features and labels
X_text = df['Review Text']
X_rating = df['Rating'].values.reshape(-1, 1)
y = df['Recommended IND']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_text_vec = vectorizer.fit_transform(X_text)

# Combine text and numeric rating
X_combined = hstack([X_text_vec, X_rating])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")