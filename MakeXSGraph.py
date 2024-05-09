import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('xs_test_set/categorized/xs_final_analysis.csv', encoding='utf-8')


fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(0, 2, 2)
width = 0.2


colors = ['#6D7973', '#7B9EA8', '#A69F7C', '#839788', '#A88DAB', '#D89C6A', '#9CAFB7', '#F4A299', '#2A4858', '#FFF1D0']
y1 = [df.iloc[2, 1]]
y2 = [df.iloc[2, 2]]
y3 = [df.iloc[2, 3]]
y4 = [df.iloc[2, 4]]
y5 = [df.iloc[2, 5]]
y6 = [df.iloc[2, 6]]
y7 = [df.iloc[2, 7]]
y8 = [df.iloc[2, 8]]
y9 = [df.iloc[2, 9]]
y10 = [df.iloc[2, 10]]

plt.bar(x-0.8, y1, width, label="Definitions", color=colors[0])
plt.bar(x-0.6, y2, width, label="Nons Group Real Discr", color=colors[1])
plt.bar(x-0.4, y3, width, label="Real Group Nons Discr", color=colors[2])
plt.bar(x-0.2, y4, width, label="Figurative Language", color=colors[3])
plt.bar(x, y5, width, label="Historical Events", color=colors[4])
plt.bar(x+0.2, y6, width, label="Homonyms", color=colors[5])
plt.bar(x+0.4, y7, width, label="Privacy Public", color=colors[6])
plt.bar(x+0.6, y8, width, label="Privacy Fiction", color=colors[7])
plt.bar(x+0.8, y9, width, label="Safe Contexts", color=colors[8])
plt.bar(x+1.0, y10, width, label="Safe Targets", color=colors[9])


ax.set_ylabel('Refusal Rate (%)', fontsize=12)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(['LLM-Guarded Model'], rotation=0, ha="center", fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=12)

plt.tight_layout()
plt.savefig('Figure2.png', format='png', dpi=300)

plt.show()