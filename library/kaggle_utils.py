import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# trainとtestで同じ値の分布に違いがあるかをDFで返す
def get_train_test_comp(train, test, col, is_viz=False, target=''):
    target_avg = train.groupby([col])[target].mean()
    cnt_train = train[col].value_counts()
    cnt_test  = test[col].value_counts()

    cnt_train.name = f"train_{col}"
    cnt_test.name  = f"test_{col}"
    df = pd.concat([cnt_train, cnt_test, target_avg], axis=1)
    df.fillna(0, inplace=True)
    df['diff'] = df[f'train_{col}'] - df[f'test_{col}']
    df['diff'] = df['diff'].map(lambda x: np.abs(x))
    df.sort_values(by=f"diff", ascending=False, inplace=True)
    df.reset_index(inplace=True)


    if is_viz:
        fig = plt.figure(figsize=(20, 4))
        fig.patch.set_facecolor('white')
        sns.set_style("whitegrid")

        # train test dist viz
        print(col)
        try:
            sns.distplot(train[col].fillna(-1))
        #     fig = plt.figure(figsize=(20, 4))
            sns.distplot(test[col].fillna(-1))
        except ValueError:
            pass

        plt.xlabel(f'{col}')
        plt.title(f'{col} Distribution', fontsize=13, alpha=0.5)
        plt.savefig(f'../kde_{col}.png')
        plt.show()

    return df

# TrainとTestにおける分布の違いをみる
def get_train_test_dist(train, test, col):
    cnt_train = train[col].value_counts()
    cnt_test  = test[col].value_counts()
    cnt_train = cnt_train.to_frame(col)
    cnt_test = cnt_test.to_frame(col)
    cnt_train['is_train'] = 1
    cnt_test['is_train'] = 0
    df = pd.concat([cnt_train, cnt_test], axis=0)
    df.reset_index(inplace=True)

    fig = plt.figure(figsize=(20, 4))
    fig.patch.set_facecolor('white')
    sns.set_style("whitegrid")

    # train test dist viz
    sns.lmplot(data=df, x='index', y=f"{col}", hue='is_train', size=14)

    plt.xlabel(f'{col}')
    plt.title(f'{col} Distribution', fontsize=13, alpha=0.5)
    plt.savefig(f'../kde_{col}.png')
    plt.show()


# カテゴリカルにおけるtargetの分布を平均で見る
def get_cat_target_dist(train, col, min_sample=10000, target=''):
    df = train.groupby(col)[target].mean().reset_index()
    cnt = train[col].value_counts()
    cnt_idx = cnt[cnt>min_sample].index
    df = df.loc[df[col].isin(cnt_idx)]

    fig = plt.figure(figsize=(20, 4))
    fig.patch.set_facecolor('white')
    sns.set_style("whitegrid")

    # train test dist viz
    sns.lmplot(data=df, x=col, y=target, size=14)

    plt.xlabel(f'{col}')
    plt.title(f'{col} Distribution', fontsize=13, alpha=0.5)
    plt.savefig(f'../kde_{col}.png')
    plt.show() 
