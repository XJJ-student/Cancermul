import pandas as pd

label_col = 'OS_time'
data = pd.read_csv('./example/status_info.csv', index_col=0)
data['OS_time'] = data['OS_time']/30
print(data)

patients_df = data
uncensored_df = patients_df[patients_df['status'] < 1]
n_bins = 4
disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
eps = 1e-6
# q_bins[0] = patients_df[label_col].min() - eps
# 更新边界值以确保覆盖所有数据范围
q_bins[0] = min(q_bins[0], patients_df[label_col].min()) - eps
q_bins[-1] = max(q_bins[-1], patients_df[label_col].max()) + eps

disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
print('disc_labels')
print(disc_labels)
print('q_bins')
print(q_bins)
patients_df.insert(2, 'label', disc_labels.values.astype(int))
print(patients_df)
patients_df.to_csv('./example/status_info_discretization.csv')
from collections import Counter
print(max(list(patients_df['OS_time'])))
print(Counter(list(patients_df['label'])))