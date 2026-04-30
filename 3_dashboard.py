import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

# Load data and models
df = pd.read_csv('spotify_cleaned.csv')
clf = joblib.load('mood_classifier.pkl')

features = ['danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'loudness', 'speechiness']

# Set style
sns.set_theme(style="whitegrid", palette="Set2")
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Music Mood & Popularity Predictor — Dashboard",
             fontsize=16, fontweight='bold', y=0.98)

# ── Chart 1 — Mood Distribution ──
ax1 = fig.add_subplot(2, 2, 1)
mood_counts = df['mood'].value_counts()
colors = ['#FAC775', '#9FE1CB', '#B5D4F4', '#F4C0D1']
bars = ax1.bar(mood_counts.index, mood_counts.values,
               color=colors, edgecolor='white', linewidth=1.5)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f'{int(bar.get_height()):,}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax1.set_title('Mood Distribution', fontsize=13, fontweight='bold')
ax1.set_xlabel('Mood Category')
ax1.set_ylabel('Number of Songs')

# ── Chart 2 — Valence vs Energy (colored by mood) ──
ax2 = fig.add_subplot(2, 2, 2)
sample = df.sample(2000, random_state=42)
mood_colors = {'Happy': '#FAC775', 'Energetic': '#9FE1CB',
               'Calm': '#B5D4F4', 'Sad': '#F4C0D1'}
for mood, group in sample.groupby('mood'):
    ax2.scatter(group['valence'], group['energy'], label=mood,
                color=mood_colors[mood], alpha=0.6, s=15, edgecolors='none')
ax2.set_title('Valence vs Energy by Mood', fontsize=13, fontweight='bold')
ax2.set_xlabel('Valence (Positivity)')
ax2.set_ylabel('Energy')
ax2.legend(title='Mood', fontsize=9)
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# ── Chart 3 — Feature Importance ──
ax3 = fig.add_subplot(2, 2, 3)
importances = pd.Series(clf.feature_importances_, index=features).sort_values()
colors_imp = ['#B5D4F4' if v < importances.max() else '#185FA5'
              for v in importances.values]
bars3 = ax3.barh(importances.index, importances.values,
                 color=colors_imp, edgecolor='white')
for bar in bars3:
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)
ax3.set_title('Feature Importance (Mood Classifier)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Importance Score')

# ── Chart 4 — Popularity vs Danceability ──
ax4 = fig.add_subplot(2, 2, 4)
sample2 = df.sample(3000, random_state=42)
ax4.scatter(sample2['danceability'], sample2['popularity'],
            alpha=0.3, s=10, color='#9FE1CB', edgecolors='none')
m, b = np.polyfit(sample2['danceability'], sample2['popularity'], 1)
x_line = np.linspace(0, 1, 100)
ax4.plot(x_line, m * x_line + b, color='#0F6E56', linewidth=2,
         label=f'y = {m:.1f}x + {b:.1f}')
ax4.set_title('Popularity vs Danceability', fontsize=13, fontweight='bold')
ax4.set_xlabel('Danceability')
ax4.set_ylabel('Popularity Score')
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Dashboard saved as dashboard.png")
