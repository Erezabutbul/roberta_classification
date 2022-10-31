import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

run_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

res_df = pd.read_csv(
    os.path.join(run_dir, 'data_full.csv'),
    on_bad_lines='skip', index_col=0)

df = pd.DataFrame()
skf = StratifiedShuffleSplit(n_splits=1, random_state=0)
# make train and test files
for train_index, val_index in skf.split(res_df["text"], res_df["labels"]):
    train_df = res_df.iloc[train_index]
    test_df = res_df.iloc[val_index]
    train_df.to_csv(
        os.path.join(run_dir, 'train_balanced.csv'),
        index=True, index_label="index", encoding="utf_8_sig")
    df = train_df
    test_df.to_csv(
        os.path.join(run_dir, 'test_balanced.csv'),
        index=True, index_label="index", encoding="utf_8_sig")
    break

n = 4
skf = StratifiedKFold(n_splits=n, random_state=5, shuffle=True)
ctr = 0
for train_index, val_index in skf.split(df["text"], df["labels"]):
    ctr += 1
    train_df_tmp = df.iloc[train_index]
    eval_df_tmp = df.iloc[val_index]
    train_str = "train_df_" + str(ctr) + ".csv"
    eval_str = "eval_df_" + str(ctr) + ".csv"
    train_df_tmp.to_csv(
        os.path.join(run_dir, 'splits', train_str), index=True, index_label="index", encoding="utf_8_sig")
    eval_df_tmp.to_csv(
        os.path.join(run_dir, 'splits', eval_str), index=True, index_label="index", encoding="utf_8_sig")
