import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset from JSON
with open("symptom_dataset.json") as f:
    data = json.load(f)

X, y = zip(*data)

# Create pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and save
model.fit(X, y)
joblib.dump(model, 'symptom_model.joblib')

print("✅ Large model trained and saved.")
