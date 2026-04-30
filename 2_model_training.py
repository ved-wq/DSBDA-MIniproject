import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load cleaned dataset
df = pd.read_csv('spotify_cleaned.csv')

features = ['danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'loudness', 'speechiness']

X = df[features]
y_mood = df['mood']
y_pop  = df['popularity']

X_train, X_test, ym_train, ym_test = train_test_split(
    X, y_mood, test_size=0.2, random_state=42)
_, _, yp_train, yp_test = train_test_split(
    X, y_pop, test_size=0.2, random_state=42)

# Mood classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, ym_train)
print("Mood Accuracy:", accuracy_score(ym_test, clf.predict(X_test)))

# Popularity regressor
reg = LinearRegression()
reg.fit(X_train, yp_train)
rmse = np.sqrt(mean_squared_error(yp_test, reg.predict(X_test)))
print("Popularity RMSE:", round(rmse, 2))

# Save models
joblib.dump(clf, 'mood_classifier.pkl')
joblib.dump(reg, 'popularity_regressor.pkl')
print("\nModels saved: mood_classifier.pkl, popularity_regressor.pkl")
