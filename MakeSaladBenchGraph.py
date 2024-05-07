import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json


df = pd.read_csv('Final Categories.csv',  encoding='utf-8')


fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(0, 8, 2)
width = 0.2



colors = ['#6D7973', '#7B9EA8', '#A69F7C', '#839788', '#A88DAB', '#D89C6A', '#9CAFB7']
y1 = [df.iloc[2, 1], df.iloc[3, 1], df.iloc[7, 1], df.iloc[8, 1]]
y2 = [df.iloc[2, 2], df.iloc[3, 2], df.iloc[7, 2], df.iloc[8, 2]]
y3 = [df.iloc[2, 3], df.iloc[3, 3], df.iloc[7, 3], df.iloc[8, 3]]
y4 = [df.iloc[2, 4], df.iloc[3, 4], df.iloc[7, 4], df.iloc[8, 4]]
y5 = [df.iloc[2, 5], df.iloc[3, 5], df.iloc[7, 5], df.iloc[8, 5]]
y6 = [df.iloc[2, 6], df.iloc[3, 6], df.iloc[7, 6], df.iloc[8, 6]]
y7 = [df.iloc[2, 7], df.iloc[3, 7], df.iloc[7, 7], df.iloc[8, 7]]

plt.bar(x-0.5, y1, width, label="Representation & Toxicity Harms", color=colors[0])
plt.bar(x-0.3, y2, width, label="Misinformation Harms", color=colors[1])
plt.bar(x-0.1, y3, width, label="Information & Safety Harms", color=colors[2])
plt.bar(x+0.1, y4, width, label="Malicious Use", color=colors[3])
plt.bar(x+0.3, y5, width, label="Human Autonomy & Integrity Harms", color=colors[4])
plt.bar(x+0.5, y6, width, label="Socioeconomic Harms", color=colors[5])
plt.bar(x+0.7, y7, width, label="Overall", color=colors[6])





ax.set_ylabel('Safety Rate (%)', fontsize=11)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(['LLM-Guarded Model', 'Base Model', 'LLM-Guarded Model', 'Base Model'], rotation=0, ha="center", fontsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=10)

text = ax.text(x[0] + 1.1, -10, 'Base Set', ha='center', va='top', fontsize=11)
ax.text(x[2] + 1.1, -10, 'Attack-enhanced Set', ha='center', va='top', fontsize=11)

plt.tight_layout()
plt.savefig('Figure1.png', format='png', dpi=300, bbox_inches='tight')

plt.show()