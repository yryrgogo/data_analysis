import sys
import os
import shutil
import datetime
import glob
import re
import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn.cluster import KMeans
from plsa_core import PLSA


column = "CATEGORY"
value = "VALUE"

# PLSA
INPUT_PLSA = '/mnt/c/Git/CCJC/dbn/input/kmeans/'
path_list = glob.glob(INPUT_PLSA + '*.csv')

# PLSAのインプットデータ
#path_list = glob.glob(OUTPUT_PLSA + '*.csv')
# for path in path_list:
#    if path.count('tate'):
#        filepath_origin = path
# filename_origin = re.search(r'/([^/.]*).csv', filepath_origin).group(1) # Linux

# kmeans


def create_kmeans(df, name, N):
        # DataFrameをListに変換
    df_arr = df.as_matrix()

    # kmeansの実行
    kmeans = KMeans(n_clusters=N, random_state=15).fit(df_arr)
    # kmeans結果
    labels = kmeans.labels_
    # 結果の付与
    df[name] = labels

    return df[name]


# データ処理
def kmeans_main(df_plsa_input, xy_top20, N):

    # PLSA後データ
    df_plsa_4  = xy_top20
    df_plsa_6  = xy_top20
    df_plsa_8  = xy_top20
    df_plsa_10 = xy_top20

    df_origin = df_plsa_input

    # 行列の作成（ピボットテーブル）
    df_table = pd.pivot_table(
        df_origin,
        values=value,
        index=[index],
        columns=[column],
        aggfunc=np.sum
    )
    df_table = df_table.fillna(0)
    df_table = df_table.sort_index()

    # kmeans
    kmeans_4  = create_kmeans(df_plsa_4 , 'plsa_4' , N)
    kmeans_6  = create_kmeans(df_plsa_6 , 'plsa_6' , N)
    kmeans_8  = create_kmeans(df_plsa_8 , 'plsa_8' , N)
    kmeans_10 = create_kmeans(df_plsa_10, 'plsa_10', N)
    kmeans_o  = create_kmeans(df_table  , 'plsa_o' , N)

    # 全kmeans結果をdf_origin（購買データ）に連結
    df_origin.index = df_origin[index]
    df_origin.drop(index, axis=1)
    df_origin = df_origin.join(kmeans_4, how='inner')
    df_origin = df_origin.join(kmeans_6, how='inner')
    df_origin = df_origin.join(kmeans_8, how='inner')
    df_origin = df_origin.join(kmeans_10, how='inner')
    df_origin = df_origin.join(kmeans_o, how='inner')

    # 出力用のフォルダ作成
    # 出力ファイルが1つのためフォルダは作成しない
#    os.mkdir("/home/gixo/plsa/output_kmeans/" + starttime)
    df_origin.to_csv(OUTPUT_PLSA + '/' + fn + '_out_plsa_kmeans{}.csv'.format(N))

    return df_origin


def plsa_main(data, dim):

    # unstackで縦持ちに直す。カラム名は

    # 行列の作成（ピボットテーブル）
    df_table = pd.pivot_table(
        df,  # 入力変数
        values=value,  # 値
        index=index,  # 行名
        columns=column,  # 列名
        aggfunc=np.sum  # 集計方法
    )
    # 結果表示用
    x = df_table.index
    y = df_table.columns
    # 確認用
    print('===============================')
    print('X : ', x.name)
    print('Y : ', y.name)
    print('===============================')
    # nanを0で置換
    n = df_table.fillna(0)
    print(df.dtypes)
    # DataFrame → 行列変換
    n_mx = n.as_matrix()

    # PLSA
    plsa = PLSA(n_mx, Z)
    plsa.train()

    # Xの出力
    #  print('P(z|x)：文書xの条件下での潜在変数zの発生確率')
    Pz_x = plsa.Px_z.T * plsa.Pz[None, :]
    Pz_x_df = pd.DataFrame(Pz_x / np.sum(Pz_x, axis=1)[:, None])
    Pz_x_df.index = x
    # ファイル書き込み
    # Pz_x_df.to_csv('/home/gixo/plsa/output_plsa/' + starttime + '/out_{}.csv'.format(x.name))
    #  Pz_x_df.to_csv(OUTPUT_PLSA + '/out_{}.csv'.format(x.name))

    # Y
    #  print('P(z|y)：単語zの条件下での潜在変数zの発生確率')
    Pz_y = plsa.Py_z.T * plsa.Pz[None, :]
    Pz_y_df = pd.DataFrame(Pz_y / np.sum(Pz_y, axis=1)[:, None])
    Pz_y_df.index = y
    # ファイルの書き込み
    # Pz_y_df.to_csv('/home/gixo/plsa/output_plsa/' + starttime + '/out_{}.csv'.format(y.name))
    #  Pz_y_df.to_csv(OUTPUT_PLSA + '/out_{}.csv'.format(y.name))

    # 確率の高い順にソートし、Top20を作成
    for i in range(0, Z):
        x_sorted = Pz_x_df.sort_values(i, ascending=False)
        x_top20 = pd.DataFrame(x_sorted.ix[:, i].head(20))
        x_top20.insert(0, 'cluster', 'c{}'.format(i))

        y_sorted = Pz_y_df.sort_values(i, ascending=False)
        y_top20 = pd.DataFrame(y_sorted.ix[:, i].head(20))
        y_top20.insert(0, 'cluster', 'c{}'.format(i))

        xy_top20 = x_top20.append(y_top20)
        #  print(xy_top20)
        # xy_top20.to_csv('/home/gixo/plsa/output_plsa/' + starttime + '/out_cl{}.csv'.format(i))
        #  xy_top20.to_csv(OUTPUT_PLSA + '/out_cl{}.csv'.format(i))

    return df, xy_top20


def plsa_kmeans(data, DIM, CLUSTER):
    df, xy_top20 = plsa_main(data, DIM)
    result = kmeans_main(df, xy_top20, CLUSTER)

    return result


if __name__ == '__main__':

    for path in path_list:
        file_path = path
        fn = re.search(r'/([^/.]*).csv', file_path).group(1)  # Linux

        for Z, N in zip(DIM, CLUSTER):
            print('次元圧縮 {}'.format(Z))
            print('クラスタ数 {}'.format(N))
            starttime = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
            # KMEANS
            OUTPUT_PLSA = "/mnt/c/Git/CCJC/dbn/output/kmeans/" + starttime
            INPUT_KMEANS = OUTPUT_PLSA + "/out_users_id.csv"
            print('出力先 {}'.format(OUTPUT_PLSA))

            # インプットファイル
            # PLSA実行結果
            filename_plsa_4 = INPUT_KMEANS
            filename_plsa_6 = INPUT_KMEANS
            filename_plsa_8 = INPUT_KMEANS
            filename_plsa_10 = INPUT_KMEANS

            main()
        shutil.move(file_path, INPUT_PLSA + "used")
