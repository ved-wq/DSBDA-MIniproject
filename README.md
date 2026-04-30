# 🎵 Music Mood & Popularity Predictor

A machine learning project that classifies songs into mood categories (Happy, Sad, Energetic, Calm) and predicts popularity scores using Spotify audio features.

## 📁 Project Structure

```
spotify_mood_predictor/
├── 1_data_preprocessing.py       # Data cleaning & mood feature engineering
├── 2_model_training.py           # Train RandomForest + LinearRegression models
├── 3_dashboard.py                # Generate 4-panel visualization dashboard
├── spotify_mood_predictor.ipynb  # Full pipeline as a Google Colab notebook
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Option A — Google Colab (Recommended)
1. Open `spotify_mood_predictor.ipynb` in Google Colab
2. Upload your `dataset.csv` when prompted
3. Run all cells in order

### Option B — Local
```bash
pip install -r requirements.txt
python 1_data_preprocessing.py   # outputs spotify_cleaned.csv
python 2_model_training.py       # outputs .pkl model files
python 3_dashboard.py            # outputs dashboard.png
```

## 📊 Features Used

| Feature | Description |
|---|---|
| `danceability` | How suitable for dancing (0–1) |
| `energy` | Intensity and activity (0–1) |
| `valence` | Musical positivity (0–1) |
| `tempo` | Beats per minute |
| `acousticness` | Acoustic confidence (0–1) |
| `loudness` | Overall loudness (dB) |
| `speechiness` | Speech presence (0–1) |

## 🎭 Mood Classification Logic

| Mood | Condition |
|---|---|
| Happy | valence > 0.6 AND energy > 0.6 |
| Sad | valence < 0.4 AND energy < 0.4 |
| Energetic | energy > 0.7 AND danceability > 0.6 |
| Calm | Everything else |

## 📈 Models

- **Mood Classifier** — `RandomForestClassifier` (100 estimators)
- **Popularity Regressor** — `LinearRegression`

## 📋 Dataset

Expects a CSV file (`dataset.csv`) with columns including:
`track_id`, `artists`, `track_genre`, `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `loudness`, `speechiness`, `popularity`

Compatible with the [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
