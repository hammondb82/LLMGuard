import pandas as pd
import json

df = pd.read_csv('attack_enchanced_set/graded/AE_graded_responses.csv')


with open('attack_enchanced_set/responses/AE_past_guard_responses.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('attack_enchanced_set/prompts/attack_enhanced_set.json', 'r', encoding='utf-8') as f:
    base_set = json.load(f)
base_df = pd.DataFrame(base_set)

# base_df = pd.read_csv('xs_test_set/prompts/xstest_v2_prompts.csv')


qid = []
for entry in data:
    qid.append(entry['aid'])

df['aid'] = qid

merged_df = pd.merge(base_df, df, on='aid', how='left')
# merged_df = merged_df.drop(['question'], axis=1)
merged_df.fillna({'labels': 0, 'class': 0}, inplace=True)

filtered_df = merged_df[merged_df['labels'] == 1]
counts_label_1 = pd.DataFrame({
    'Count_Where_Label_1': filtered_df[['1-category', '2-category', '3-category']].apply(pd.Series.value_counts).sum(axis=1)
})
total_counts = pd.DataFrame({
    'Total_Counts': merged_df[['1-category', '2-category', '3-category']].apply(pd.Series.value_counts).sum(axis=1)
})
counts_df = counts_label_1.join(total_counts, how='outer').fillna(0)
counts_df.index.name = 'Category'
csv_file_path = 'attack_enchanced_set/categorized/AE_category_counts.csv'
counts_df.to_csv(csv_file_path)