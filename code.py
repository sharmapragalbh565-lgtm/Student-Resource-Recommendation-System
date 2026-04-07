import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

df = pd.read_csv('vit_bhopal_students_dataset.csv')

def to_set(text):
    return set(map(str.strip, str(text).split(',')))

def generate_pairs(df, n_pairs=1000):
    pairs = []
    indices = df.index.tolist()
    while len(pairs) < n_pairs:
        a, b = random.sample(indices, 2)
        topics_a, topics_b = to_set(df.loc[a, 'Subjects']), to_set(df.loc[b, 'Subjects'])
        resources_a, resources_b = to_set(df.loc[a, 'Syllabus_Topics']), to_set(df.loc[b, 'Syllabus_Topics'])

        match = int(bool(topics_a & topics_b and resources_a & resources_b))

        pairs.append((a, b, match))
    return pairs

def extract_features(a, b):
    topics_a, topics_b = to_set(df.loc[a, 'Subjects']), to_set(df.loc[b, 'Subjects'])
    resources_a, resources_b = to_set(df.loc[a, 'Syllabus_Topics']), to_set(df.loc[b, 'Syllabus_Topics'])

    return [
        len(topics_a & topics_b),
        len(resources_a & resources_b)
    ]

pairs = generate_pairs(df, 1000)
X = [extract_features(a, b) for a, b, _ in pairs]
y = [label for _, _, label in pairs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Resource Recommender Accuracy:", accuracy)

def recommend_resources(student_id, top_n=5):
    idx = df.index[df['Student_ID'] == student_id].tolist()[0]
    scores = []
    for other in df.index:
        if other == idx:
            continue
        features = extract_features(idx, other)
        prob = clf.predict_proba([features])[0][1]
        scores.append((other, prob))
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return df.loc[[i for i, _ in top_matches]][['Student_ID', 'Syllabus_Topics', 'Subjects']]

# Sample recommendation
sample_id = df['Student_ID'].iloc[0]
print("Top resource recommendations for:", sample_id)
print(recommend_resources(sample_id))
