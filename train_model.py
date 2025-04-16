from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data
texts = [
    "I love this movie",
    "This is a great product",
    "I hate this thing",
    "This was the worst experience ever",
    "Amazing service and friendly staff",
    "Terrible food, will not come back"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train model
model.fit(texts, labels)

# Save model
joblib.dump(model, 'sentimental_model.pkl')

print("Model saved as sentimental_model.pkl ✅")
