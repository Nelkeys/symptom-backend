import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load CSV
df = pd.read_csv("symptom_dataset.csv")  # Replace with actual file name

# Step 2: Extract input/output
X = df["text"].values
y = df["label"].values

# Step 3: Build Pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Step 4: Train and Save
model.fit(X, y)
joblib.dump(model, 'symptom_model.joblib')

print("✅ Model trained and saved.")
