# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys


input_path = "../input/tabelog_test_train.csv"
output_path = "../output/tabelog_genre_list.csv"

genre_list= [
    'izakaya',
    'ita_fre',
    'bal',
    'dinning',
    'meat',
    'sushi',
    'japan',
    'nabe',
    'china',
    'countries',
    'ramen',
    'light',
    'cook'
]

genre_dict = {
    'izakaya':[
    '居酒屋',
    '焼鳥',
    '串揚げ',
    '串かつ',
    '串焼き',
    '沖縄',
    '魚',
    '海鮮'
    ],
    'ita_fre':[
        'イタリアン',
        'パスタ',
        'ピザ',
        '西洋',
        'フレンチ'
    ],
    'bal':[
        'バル',
        'バール',
        'ビストロ',
        'スペイン',
        '地中海'
    ],
    'dinning':[
        'ダイニングバー',
        'ダイニング',
        'バー',
        'ビア',
        'アメリカ',
        'ドイツ',
        'ラウンジ'
    ],
    'meat':[
        '焼肉',
        'ホルモン',
        'ジンギスカン',
        '肉'
    ],
    'sushi':['寿司'],
    'japan':[
        '和食',
        '割烹',
        '懐石',
        '会席',
        '郷土',
        'うなぎ',
        'ふぐ',
        'かに',
        'ろばた',
        '天ぷら'
    ],
    'nabe':[
        '鍋',
        'もつ鍋',
        'しゃぶしゃぶ',
        'すきやき',
        'すき焼き',
        '水炊き'
    ],
    'china':[
        '中華',
        '餃子',
        '韓国',
        '四川',
        '広東',
        '台湾',
        '中国',
        '上海'
    ],
    'countries':[
        'カレー',
        'インド',
        '各国',
        'アジア',
        'エスニック',
        'メキシコ',
        'ベトナム',
        '南米',
        'シンガポール',
        'ネパール',
        'トルコ',
        'ブラジル',
        'パキスタン'
    ],
    'ramen':[
        'ラーメン',
        'つけ麺',
        '麺'
    ],
    'light':[
        'そば',
        'うどん',
        '洋食',
        '丼',
        'ステーキ',
        'お好み焼き',
        'ハンバーグ'
    ],
    'cook':[
        '創作料理',
        '鳥料理',
        '鉄板焼き',
        '牛料理'
    ]
}


def spt(x): return x.split("、")


def genre_trans(df): 
    df.genre = df.genre.astype('str')
    return df['genre'].apply(lambda x:pd.Series(spt(x) if len(spt(x))==3 else spt(x)+['0'] if len(spt(x))==2 else spt(x)+['0']*2 if len(spt(x))==1 else spt(x) ['0']*3 if len(spt(x))==0 else [1,2,3]))


def extract_genre_list():

    data = pd.read_csv(input_path)
    print(data.head())

    genre_data = genre_trans(data)

    a2 = np.append(np.append(genre_data[0].values, genre_data[1].values), genre_data[2].values)
    u, c = np.unique(a2, return_counts=True)
    genre_dict = dict(zip(u, c))
    df = pd.Series(genre_dict).sort_values(ascending=False)
    print(df.head(10))

    df.to_csv(output_path, index=True, encoding="utf-8")
    
    
def genre_label(x):
    for genre in genre_list:
        for check in genre_dict[genre]:
            if x.count(check):
                return genre
    return 'other'

    
def genre_dummies():
    
    data = pd.read_csv(input_path)
    genre_data = genre_trans(data)
    df = pd.concat([data.name, genre_data], axis=1)
    
    for i in range(3):
        if i == 0:tmp = df[['name', i]].rename(columns={i:'genre'})
        else :
            tmp2 = df[['name', i]].rename(columns={i:'genre'})    
            tmp  = pd.concat([tmp, tmp2[['name', 'genre']]], axis=0)
    
    tmp.fillna('0', inplace=True)
    
    label = tmp.apply(lambda x: genre_label(x[1]), axis=1)
    
    tmp_result = pd.concat([tmp.name, label], axis=1).drop_duplicates().reset_index(drop=True)
    
    # 各ジャンルの出現回数をカウント
    u, c = np.unique(np.array(label), return_counts=True)
    tmp_cnt   = dict(zip(u, c))
    tmp_cnt['other'] = 30 # otherは中の1ジャンルあたりの平均出現回数にする
    genre_cnt = pd.Series(tmp_cnt)
    genre_cnt = genre_cnt.reset_index().rename(columns={'index':'genre'})
    
    tmp_result.rename(columns={0:'genre'}, inplace=True)
    tmp_result = tmp_result.merge(genre_cnt, on='genre', how='inner')
    
    # ダミー変数化
    dummies = pd.get_dummies(tmp_result.genre)
    
    result= pd.concat([tmp_result, dummies], axis=1)
    
    # 最終的な説明変数を作成
    columns = list(result.columns) # genre カラムを削除
    columns.pop(1)
    explain = result[columns].groupby('name', as_index=False).sum()
    explain.rename(columns={0:'genre_cnt'}, inplace=True)
    
    return explain
    
    
if __name__ =='__main__':

    main()

