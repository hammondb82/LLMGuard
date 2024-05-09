import pandas as pd
import json
df = pd.read_csv('attack_enchanced_set/graded/AE_graded_responses.csv')
with open('attack_enchanced_set/prompts/attack_enhanced_set.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
json_df = pd.DataFrame(data)
df = df.join(json_df[['1-category', '2-category', '3-category']])
filtered_df = df[df['labels'] == 1]
counts_label_1 = pd.DataFrame({
    'Count_Where_Label_1': filtered_df[['1-category', '2-category', '3-category']].apply(pd.Series.value_counts).sum(axis=1)
})
total_counts = pd.DataFrame({
    'Total_Counts': df[['1-category', '2-category', '3-category']].apply(pd.Series.value_counts).sum(axis=1)
})
counts_df = counts_label_1.join(total_counts, how='outer').fillna(0)
counts_df.index.name = 'Category'
csv_file_path = 'attack_enchanced_set/categorized/AE_category_counts.csv'
counts_df.to_csv(csv_file_path)