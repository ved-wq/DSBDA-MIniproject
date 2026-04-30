import pandas as pd
import numpy as np

# Step 1 — Load dataset
df = pd.read_csv('dataset.csv')
print("Shape:", df.shape)

# Step 2 — Identify missing/null data
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\nNull % per column:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Step 3 — Handle missing data
# Drop rows where key audio features are null
df.dropna(subset=['danceability', 'energy', 'valence', 'tempo'], inplace=True)

# Fill remaining numeric nulls with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical nulls with mode
df['track_genre'].fillna(df['track_genre'].mode()[0], inplace=True)
df['artists'].fillna('Unknown', inplace=True)

# Step 4 — Remove duplicates
df.drop_duplicates(subset=['track_id'], inplace=True)

# Step 5 — Feature engineering — add Mood label
def assign_mood(row):
    if row['valence'] > 0.6 and row['energy'] > 0.6:
        return 'Happy'
    elif row['valence'] < 0.4 and row['energy'] < 0.4:
        return 'Sad'
    elif row['energy'] > 0.7 and row['danceability'] > 0.6:
        return 'Energetic'
    else:
        return 'Calm'

df['mood'] = df.apply(assign_mood, axis=1)

# Step 6 — Save cleaned sheet
df.to_csv('spotify_cleaned.csv', index=False)
print("\nCleaned dataset saved. Shape:", df.shape)
print("\nMood distribution:\n", df['mood'].value_counts())
