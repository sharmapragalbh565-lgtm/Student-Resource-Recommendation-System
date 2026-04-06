# Student-Resource-Recommendation-System
Overview:

The Student Resource Recommendation System is a Machine Learning-based system that recommends relevant study resources and peer connections based on subject similarity and syllabus topic overlap.

This system helps students:

Find relevant study materials

Connect with students studying similar subjects

Improve collaborative learning

Enhance academic performance

# Features

Machine Learning-based recommendation

Peer matching system

Resource similarity detection

Personalized recommendations

Random Forest Classification model

# Technologies Used

Python

Pandas

Scikit-learn

Random Forest Algorithm

Machine Learning

# Machine Learning Model

Model Used: Random Forest Classifier

# Purpose:

Predict similarity between students based on:

Subject overlap

Syllabus topic overlap

# How It Works
Step 1: Load Dataset

Dataset contains:

Student ID

Subjects

Syllabus Topics

Step 2: Generate Student Pairs

Pairs of students are created and labeled based on similarity.

Match = 1 if:

Subjects overlap
AND

Syllabus topics overlap

Step 3: Feature Extraction

Features used:

Number of common subjects

Number of common syllabus topics

Step 4: Train Random Forest Model

Model learns similarity patterns between students.

Step 5: Recommend Resources

Model predicts similarity probability and recommends top matching students.

# Example Output
Recommendations for Student: VIT2023001

Student_ID    Subjects             Syllabus_Topics
VIT2023010    ML, AI               Regression, CNN
VIT2023025    AI, Data Science     NLP, Classification
VIT2023042    ML, Python           Neural Networks

# Project Structure
Student-Resource-Recommender/
│
├── vit_bhopal_students_dataset.csv
├── recommender.py
├── README.md
└── requirements.txt

# Installation
pip install pandas scikit-learn
Run the Project
python recommender.py
Model Accuracy

Typical Accuracy: 85% – 95% (depends on dataset)

# Applications

University learning platforms

Peer learning systems

EdTech platforms

Study group recommendation

Smart education systems

# Future Improvements

Deep Learning recommendation model

Real-time recommendation system

Web application integration

Collaborative filtering
