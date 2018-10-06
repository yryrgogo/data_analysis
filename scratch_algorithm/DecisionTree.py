import numpy as np
import pandas as pd


def gini_score(data, target, feat_idx, threshold):
    gini = 0
    sample_num = len(target)

    # TargetをThresholdで分割し2つの子ノードを格納する
    div_target = [target[data[:, feat_idx] >= threshold], target[data[:, feat_idx] < threshold]]

    # 子ノード(分割されたTarget Group)を取り出す
    for group in div_target:
        score = 0
        label_list = np.unique(group) # 子ノード内のユニークラベル
        for label in label_list:
            # 子ノード内におけるそのlabelの割合
            p = np.sum(group==label)/len(group)
            score += p*p
        # そのノードの不純度は、分かれた子ノード2つのジニ計数の加重平均にする
        gini += (1-score) * (len(group)/sample_num)
    return gini


def search_best_split(data, target):
