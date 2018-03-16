import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from itertools import combinations
import datetime
import sys
import pickle
import plsa_core
from plsa_kmeans import plsa_kmeans
#  , plsa_main, kmeans_main

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

output_path = '../input/' + start_time + '_ncaa_submit_feature.csv'
sample_path = '../input/2018/SampleSubmissionStage2_SampleTourney2018.csv'
sample = pd.read_csv(sample_path)
sample['season'] = sample.apply(lambda x:int(x.ID.split('_')[0]), axis=1)
sample['teamid'] = sample.apply(lambda x:int(x.ID.split('_')[1]), axis=1)
sample['teamid_2'] = sample.apply(lambda x:int(x.ID.split('_')[2]), axis=1)
sample.drop('Pred', axis=1, inplace=True)
submit_base = sample


# preprocessing****************************
# load_data********************************
#  df = pd.read_csv('../input/20180314_2018__feature_for_submit.csv')
#  df = pd.read_csv('../input/20180315_223852_ncaa_feature_set.csv')
df = pd.read_csv('../input/20180315_ncaa_compact_sql_feature.csv')
#  print(df.head())
#  sys.exit()
#  a = sample[['teamid', 'season']].drop_duplicates()
#  b = df[['teamid', 'season']].drop_duplicates()
#  c = a.merge(b, on= ['teamid', 'season'], how='inner')
#  print(c['teamid'].drop_duplicates().count())
#  print(c.count())
#  print(c.head())
#  print(df[df['season']==2018]['teamid'].drop_duplicates().count())
#  sys.exit()

# feature_setting
#  season_list = np.arange(2014, 2019, 1)
season_list = [2018]
day_list = [133]
this_season_feature = ['result', 'score', 'score_diff', 'numot'
                       #  ,
                       #  'ordinalrank',
                       #  'ordinalrank_diff',
                       #  'fgm',
                       #  'fga',
                       #  'fgm3',
                       #  'fga3',
                       #  'ftm',
                       #  'fta',
                       #  'or_',
                       #  'dr',
                       #  'ast',
                       #  'to_',
                       #  'stl',
                       #  'blk',
                       #  'pf',
                       #  'fgm_diff',
                       #  'fga_diff',
                       #  'fgm3_diff',
                       #  'fga3_diff',
                       #  'ftm_diff',
                       #  'fta_diff',
                       #  'or_diff',
                       #  'dr_diff',
                       #  'ast_diff',
                       #  'to_diff',
                       #  'stl_diff',
                       #  'blk_diff',
                       #  'pf_diff'
                       ]

# 各シーズンでncaa tournament の参加チームリストを作る

# score_diff
df['score_diff'] = df['score'] - df['score_2']
#  df['fgm_diff'] = df['fgm'] - df['fgm_2']
#  df['fga_diff'] = df['fga'] - df['fga_2']
#  df['fgm3_diff'] = df['fgm3'] - df['fgm3_2']
#  df['fga3_diff'] = df['fga3'] - df['fga3_2']
#  df['ftm_diff'] = df['ftm'] - df['ftm_2']
#  df['fta_diff'] = df['fta'] - df['fta_2']
#  df['or_diff'] = df['or_'] - df['or_2']
#  df['dr_diff'] = df['dr'] - df['dr_2']
#  df['ast_diff'] = df['ast'] - df['ast_2']
#  df['to_diff'] = df['to_'] - df['to_2']
#  df['stl_diff'] = df['stl'] - df['stl_2']
#  df['blk_diff'] = df['blk'] - df['blk_2']
#  df['pf_diff'] = df['pf'] - df['pf_2']
#  df['ordinalrank_diff'] = df['ordinalrank'] - df['ordinalrank_2']

df_away = df[df.location == 'A']  # away-FE
df_home = df[df.location == 'H']  # home-FE
df_neutral = df[df.location == 'N']  # neutral-FE

# season aggrigation はmean以外もやる？

# win_percent-FE
df_wp = df.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
    'result'].mean().rename(columns={'result': 'win_percent_season'})
#  df_wp_away = df_away.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].mean().rename(columns={'result': 'win_percent_season_away'})
#  df_wp_home = df_home.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].mean().rename(columns={'result': 'win_percent_season_home'})
#  df_wp_neutral = df_neutral.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].mean().rename(columns={'result': 'win_percent_season_neutral'})

# score-FE
# seasonあたりのscore統計量を求めるため、seasonでgroupbyしておく
df_score = df.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
    'score'].mean().rename(columns={'score': 'score_season'})
#  df_score_away = df_away.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score'].mean().rename(columns={'score': 'score_season_away'})
#  df_score_home = df_home.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score'].mean().rename(columns={'score': 'score_season_home'})
#  df_score_neutral = df_neutral.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score'].mean().rename(columns={'score': 'score_season_neutral'})

# score_diff-FE
# seasonあたりのscore統計量を求めるため、seasonでgroupbyしておく
df_score_diff = df.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
    'score_diff'].mean().rename(columns={'score_diff': 'score_diff_season'})
#  df_score_diff_away = df_away.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score_diff'].mean().rename(columns={'score_diff': 'score_diff_season_away'})
#  df_score_diff_home = df_home.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score_diff'].mean().rename(columns={'score_diff': 'score_diff_season_home'})
#  df_score_diff_neutral = df_neutral.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'score_diff'].mean().rename(columns={'score_diff': 'score_diff_season_neutral'})

# match_number-FE
#  df_match = df.drop_duplicates().groupby(['teamid', 'firstd1season', 'season'])[
#      'daynum'].size().reset_index().rename(columns={'daynum': 'match_season'})
#  df_match_away = df_away.drop_duplicates().groupby(['teamid', 'firstd1season', 'season'])[
#      'daynum'].size().reset_index().rename(columns={'daynum': 'match_season_away'})
#  df_match_home = df_home.drop_duplicates().groupby(['teamid', 'firstd1season', 'season'])[
#      'daynum'].size().reset_index().rename(columns={'daynum': 'match_season_home'})
#  df_match_neutral = df_neutral.drop_duplicates().groupby(['teamid', 'firstd1season', 'season'])[
#      'daynum'].size().reset_index().rename(columns={'daynum': 'match_season_neutral'})

# win_num-FE
df_win = df.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
    'result'].sum().rename(columns={'result': 'win_num_season'})
#  df_win_away = df_away.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].sum().rename(columns={'result': 'win_num_season_away'})
#  df_win_home = df_home.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].sum().rename(columns={'result': 'win_num_season_home'})
#  df_win_neutral = df_neutral.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'result'].sum().rename(columns={'result': 'win_num_season_neutral'})


# numot-FE
#  df_ot = df.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'numot'].sum().rename(columns={'numot': 'ot_season'})
#  df_ot_away = df_away.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'numot'].sum().rename(columns={'numot': 'ot_season_away'})
#  df_ot_home = df_home.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'numot'].sum().rename(columns={'numot': 'ot_season_home'})
#  df_ot_neutral = df_neutral.groupby(['teamid', 'firstd1season', 'season'], as_index=False)[
#      'numot'].sum().rename(columns={'numot': 'ot_season_neutral'})


feature_df = [
    #  df,
    #  df,
    df_wp,
    #  df_wp_away,
    #  df_wp_home,
    #  df_wp_neutral,
    df_score,
    #  df_score_away,
    #  df_score_home,
    #  df_score_neutral,
    df_score_diff,
    #  df_score_diff_away,
    #  df_score_diff_home,
    #  df_score_diff_neutral,
    #  df_match,
    #  df_match_away,
    #  df_match_home,
    #  df_match_neutral,
    df_win
    #  ,
    #  df_win_away,
    #  df_win_home,
    #  df_win_neutral,
    #  df_ot
    #  ,
    #  df_ot_away,
    #  df_ot_home,
    #  df_ot_neutral
]

feature_name = [
    #  'seed_flg',
    #  'seed_num',
    'win_percent_season',
    #  'win_percent_season_away',
    #  'win_percent_season_home',
    #  'win_percent_season_neutral',
    'score_season',
    #  'score_season_away',
    #  'score_season_home',
    #  'score_season_neutral',
    'score_diff_season',
    #  'score_diff_season_away',
    #  'score_diff_season_home',
    #  'score_diff_season_neutral',
    #  'match_season',
    #  'match_season_away',
    #  'match_season_home',
    #  'match_season_neutral',
    'win_num_season'
    #  ,
    #  'win_num_season_away',
    #  'win_num_season_home',
    #  'win_num_season_neutral',
    #  'ot_season'
    #  ,
    #  'ot_season_away',
    #  'ot_season_home',
    #  'ot_season_neutral'
]

# 基本統計量-直近3シーズン- 引数は今シーズン
# this_seasonより前のシーズンで集計


def base_stats_3season(data, this_season, value):
    tmp = data[(data['season'] >= this_season-3) & (data['season']
                                                    < this_season)][['teamid', 'firstd1season', 'season', value]].drop_duplicates()

    if value == 'seed_flg' or value.count('ot_season'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_avg_3season'.format(value): 'mean',
                                                                     '{}_var_3season'.format(value): 'var'})
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_3season'.format(value): 'sum',
                                                                     '{}_avg_3season'.format(value): 'mean',
                                                                     '{}_max_3season'.format(value): 'max',
                                                                     '{}_median_3season'.format(value): 'median',
                                                                     '{}_min_3season'.format(value): 'min',
                                                                     '{}_var_3season'.format(value): 'var'})

    result.fillna(-1, inplace=True)  # varは成分が一つだとNaNになるため
    # 3season分の実績をもたないチームの値は全て-1とする
    result.set_index('teamid', inplace=True)
    columns = result.columns
    young_team = list(
        tmp[tmp['firstd1season'] > this_season-3]['teamid'].drop_duplicates())
    result.loc[young_team, columns] = -1
    return result.reset_index()


# 基本統計量-直近6シーズン- 引数は今シーズン
# this_seasonより前のシーズンで集計
def base_stats_6season(data, this_season, value):
    tmp = data[(data['season'] >= this_season-6) & (data['season'] < this_season)
               ][['teamid', 'firstd1season', 'season', value]].drop_duplicates()

    if value == 'seed_flg' or value.count('ot_season'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_avg_6season'.format(value): 'mean',
                                                                     '{}_var_6season'.format(value): 'var'})
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_6season'.format(value): 'sum',
                                                                     '{}_avg_6season'.format(value): 'mean',
                                                                     '{}_max_6season'.format(value): 'max',
                                                                     '{}_median_6season'.format(value): 'median',
                                                                     '{}_min_6season'.format(value): 'min',
                                                                     '{}_var_6season'.format(value): 'var'})

    result.fillna(-1, inplace=True)  # varは成分が一つだとNaNになるため
# 6season分の実績をもたないチームの値は全て-1とする
    result.set_index('teamid', inplace=True)
    columns = result.columns
    young_team = list(
        tmp[tmp['firstd1season'] > this_season-6]['teamid'].drop_duplicates())
    result.loc[young_team, columns] = -1
    return result.reset_index()


# 基本統計量-過去シーズン全て
# this_seasonより前のシーズンで集計
def base_stats_all(data, this_season, value):
    tmp = data[data['season'] < this_season][[
        'teamid', 'season', value]].drop_duplicates()
    if value == 'seed_flg' or value.count('ot_season'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_avg_all'.format(value): 'mean',
                                                                     '{}_var_all'.format(value): 'var'})
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_all'.format(value): 'sum',
                                                                     '{}_avg_all'.format(value): 'mean',
                                                                     '{}_max_all'.format(value): 'max',
                                                                     '{}_median_all'.format(value): 'median',
                                                                     '{}_min_all'.format(value): 'min',
                                                                     '{}_var_all'.format(value): 'var'})
    result.fillna(-1, inplace=True)
    return result


# 重み付き平均-dataに含まれるシーズン全て-dataの直近シーズンがendになる
# this_seasonより前のシーズンで集計
def weight_avg_all(data, this_season, ratio, value):
    tmp = data[data['season'] < this_season][[
        'teamid', 'season', value]].drop_duplicates()
    tmp['num'] = np.abs(tmp['season'] - (this_season-1))
    tmp['weight'] = tmp['num'].map(lambda x: ratio**x)
    tmp[value] = tmp[value]*tmp['weight']
    result = tmp.groupby(['teamid'], as_index=False)[value, 'weight'].sum()
    result['{}_all_wmean'.format(value)] = result[value] / result['weight']
    result.drop([value, 'weight'], axis=1, inplace=True)
    return result


# dataにある最新シーズンからみた1~3シーズン前のfeatureをJOIN
# 直近3シーズンのfeatureをJOIN
def feature_3season_ago(data, value):
    tmp = data[['teamid', 'season', value]].drop_duplicates().sort_values(by=[
        'teamid', 'season'])
    tmp_1ago = tmp.apply(lambda x: pd.Series([x.teamid, x.season+1, x[value]]), axis=1).rename(
        columns={0: 'teamid', 1: 'season', 2: '{}_1ago'.format(value)})
    tmp_2ago = tmp.apply(lambda x: pd.Series([x.teamid, x.season+2, x[value]]), axis=1).rename(
        columns={0: 'teamid', 1: 'season', 2: '{}_2ago'.format(value)})
    tmp_3ago = tmp.apply(lambda x: pd.Series([x.teamid, x.season+3, x[value]]), axis=1).rename(
        columns={0: 'teamid', 1: 'season', 2: '{}_3ago'.format(value)})

# 1~3season ago join
    tmp = tmp.merge(tmp_1ago, on=['teamid', 'season'], how='left')
    tmp = tmp.merge(tmp_2ago, on=['teamid', 'season'], how='left')
    result = tmp.merge(tmp_3ago, on=['teamid', 'season'], how='left')

# 1~3season ago compare
    if value == 'seed_flg' or value.count('ot_season'):
        result.fillna(-1, inplace=True)
        return result
    else:
        result['{}_ratio_1_2'.format(value)] = result['{}_1ago'.format(
            value)] / result['{}_2ago'.format(value)]
        result['{}_ratio_2_3'.format(value)] = result['{}_2ago'.format(
            value)] / result['{}_3ago'.format(value)]
    result.fillna(-1, inplace=True)
    return result


# 基本統計量-引数は今シーズンと集計したい時点のdaynum。nowを含めずに集計する
def base_stats_this_season_all(data, now, value):
    tmp = data[data['daynum'] < now]
    if value.count('score') or value.count('rank'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_all'.format(value): 'sum',
                                                                     '{}_avg_this_season_all'.format(value): 'mean',
                                                                     '{}_max_this_season_all'.format(value): 'max',
                                                                     '{}_min_this_season_all'.format(value): 'min',
                                                                     '{}_var_this_season_all'.format(value): 'var',
                                                                     })
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_all'.format(value): 'sum',
                                                                     '{}_avg_this_season_all'.format(value): 'mean'})
    result.fillna(-1, inplace=True)
    return result


def base_stats_this_season_3match(data, now, value):
    tmp = data[(data['daynum'] < now) & (data['daynum'] >= now-10)]
    if value.count('score') or value.count('rank'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_3match'.format(value): 'sum',
                                                                     '{}_avg_this_season_3match'.format(value): 'mean',
                                                                     '{}_max_this_season_3match'.format(value): 'max',
                                                                     '{}_min_this_season_3match'.format(value): 'min',
                                                                     '{}_var_this_season_3match'.format(value): 'var',
                                                                     })
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_3match'.format(value): 'sum',
                                                                     '{}_avg_this_season_3match'.format(value): 'mean'})

    result.fillna(-1, inplace=True)
    return result


def base_stats_this_season_6match(data, now, value):
    tmp = data[(data['daynum'] < now) & (data['daynum'] >= now-20)]
    if value.count('score') or value.count('rank'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_6match'.format(value): 'sum',
                                                                     '{}_avg_this_season_6match'.format(value): 'mean',
                                                                     '{}_max_this_season_6match'.format(value): 'max',
                                                                     '{}_min_this_season_6match'.format(value): 'min',
                                                                     '{}_var_this_season_6match'.format(value): 'var',
                                                                     })
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_6match'.format(value): 'sum',
                                                                     '{}_avg_this_season_6match'.format(value): 'mean'})

    result.fillna(-1, inplace=True)
    return result


def base_stats_this_season_9match(data, now, value):
    tmp = data[(data['daynum'] < now) & (data['daynum'] >= now-30)]
    if value.count('score') or value.count('rank'):
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_9match'.format(value): 'sum',
                                                                     '{}_avg_this_season_9match'.format(value): 'mean',
                                                                     '{}_max_this_season_9match'.format(value): 'max',
                                                                     '{}_min_this_season_9match'.format(value): 'min',
                                                                     '{}_var_this_season_9match'.format(value): 'var',
                                                                     })
    else:
        result = tmp.groupby(['teamid'], as_index=False)[value].agg({'{}_sum_this_season_9match'.format(value): 'sum',
                                                                     '{}_avg_this_season_9match'.format(value): 'mean'})

    result.fillna(-1, inplace=True)
    return result


# シード番号はセレクションサンデー以降でないとわからない
def seed_info_this_season(data, this_season, now, value):
    tmp = data[data.season == this_season]
    result = tmp[tmp['daynum'] <= now][['teamid', value]].drop_duplicates()
    return result


# ②直近3seasonの特徴量でクラスタリング
def team_rank1(data, DIM, CLUETER):
    data.set_index('teamid', drop=True, inplace=True)
    df_stack = data.stack().reset_index().rename(
        columns={'level_1': 'CATEGORY', 0: 'VALUE'})
    print(df_stack.head(30))
    result = plsa_kmeans(df_stack, DIM, CLUSTER, 'teamid', 'CATEGORY', 'VALUE')
    return 0


# ③直近3seasonの各チームに対するスコア合計を使ってクラスタリング(勝敗verも作ってみて比較はしたい)
def team_rank2(data):
    #  tmp = plsa(data)
    #  result = kmeans(tmp)
    return 0


# ④⑤rank1を比較して、格上に勝ったら1、負け試合は0。同レベルは0。格下に買っても0負けたら-1
def dark_horse(data, this_season):
    tmp = data[data['season'] < this_season][['teamid', 'season',
                                              'daynum', 'rank1', 'rank2', 'result']].drop_duplicates()
    tmp['dark_horse'] = data.apply(lambda x: 0 if (
        x.result == 0 or x.rank1 >= x.rank2) else 1, axis=1)
    result = tmp.groupby(['teamid', 'season'], as_index=False).agg({'darak_horse_avg_{}'.format(this_season): 'mean',
                                                                    'darak_horse_var_{}'.format(this_season): 'var',
                                                                    })
    return result

# 今シーズン-FE


def this_season_feature_set():
    for h, location in enumerate(['all']):
        if location.count('all'):
            tmp_df = df
        if location.count('away'):
            tmp_df = df_away
        if location.count('home'):
            tmp_df = df_home
        if location.count('neutral'):
            tmp_df = df_neutral
        for i, season in enumerate(season_list):
            tmp_f = tmp_df[tmp_df['season'] == season]
            for j, day in enumerate(day_list):
                tmp_match = tmp_df[tmp_df['season'] == season][[
                    'teamid', 'season']].drop_duplicates()
                tmp_match['daynum'] = day
                #  print(tmp_match['teamid'].count())
                #  sys.exit()
                for key in this_season_feature:
                    if key.count('seed') or key.count('match') or key.count('day'):
                        continue

                    if not(location.count('all')):
                        f0 = base_stats_this_season_all(tmp_f, day, key)
                        tmp_match = tmp_match.merge(
                            f0, on='teamid', how='left')
                        tmp_match.fillna(-1, inplace=True)
                    else:
                        f0 = base_stats_this_season_all(tmp_f,    day, key)
                        f1 = base_stats_this_season_3match(tmp_f, day, key)
                        f2 = base_stats_this_season_6match(tmp_f, day, key)
                        f3 = base_stats_this_season_9match(tmp_f, day, key)
                        for f in [f0, f1, f2, f3]:
                            tmp_match = tmp_match.merge(
                                f, on='teamid', how='left')
                            tmp_match.fillna(-1, inplace=True)
                if j == 0:
                    match_f = tmp_match
                else:
                    match_f = pd.concat([match_f, tmp_match], axis=0)

            if i == 0:
                this_season_f = match_f
            else:
                this_season_f = pd.concat([this_season_f, match_f], axis=0)

        if not(location.count('all')):
            for col in this_season_f.columns:
                if col == 'teamid' or col == 'season' or col == 'match_no' or col == 'daynum':
                    continue
                this_season_f.rename( columns={col: col+'_'+location}, inplace=True)
        if h == 0:
            this_season_f_set = this_season_f
        else:
            this_season_f_set = this_season_f_set.merge(this_season_f, on=['teamid', 'season', 'daynum'], how='left')

    this_season_f_set.fillna(-1, inplace=True)
    return this_season_f_set


# past season aggrigation feature
for i, season in enumerate(season_list):
    tmp_base = df[df['season'] == season][[
        'teamid', 'season']].drop_duplicates()
    for j, elem, key in zip(np.arange(len(feature_name)), feature_df, feature_name):
        f1 = base_stats_3season(elem, season, key)
        #  f2 = base_stats_6season(elem, season, key)
        f3 = base_stats_all(elem, season, key)
        f4 = weight_avg_all(elem, season, 0.8, key)
        #  for f in [f1, f2, f3, f4]:
        for f in [f1, f3, f4]:
            tmp_base = tmp_base.merge(f, on='teamid', how='left')

    if i == 0:
        past_agg_fe = tmp_base
    else:
        past_agg_fe = pd.concat([past_agg_fe, tmp_base], axis=0)

# これで全Teamのlast-3season-feartureがつく
#  for i, elem, key in zip(np.arange(len(feature_name)), feature_df, feature_name):
#      tmp_past = feature_3season_ago(elem, key)
#      if i == 0:
#          past_fe = tmp_past
#      else:
#          past_fe = past_fe.merge(tmp_past, on=['teamid', 'season'], how='inner')

# 今シーズン-FE
this_season_fe = this_season_feature_set()

# 今シーズンシード有無とシード番号。Daynum132で分かるから、133以降も試合がある＝seedとしてfeatureをつける
#  this_season_seed = df[df['daynum'] > 132][['teamid', 'season', 'seed_flg', 'seed_num']].drop_duplicates().rename(columns={'seed_flg': 'seed_flg_this_season', 'seed_num': 'seed_num_this_season'})

prime = df[['teamid', 'coachname',  'firstd1season', 'season']].drop_duplicates()

#  print(past_agg_fe.count())
#  print(this_season_fe.count())
#  print(prime.count())
#  sys.exit()
#  fe_set = past_agg_fe.merge(past_fe, on=['teamid', 'season'], how='inner')
fe_set = past_agg_fe
fe_set = fe_set.merge(this_season_fe, on=['teamid', 'season'], how='inner')
fe_set = fe_set.merge(prime, on=['teamid', 'season'], how='inner')
fe_set.fillna(-1, inplace=True)

lose_team_fe = fe_set.copy()
#  lose_team_fe = fe_set.copy().drop(['teamid_2'], axis=1)
#  lose_team_fe = lose_team_fe.drop_duplicates()
for col in lose_team_fe.columns:
    if col == 'season':
        continue
    lose_team_fe.rename(columns={col: col+'_2'}, inplace=True)

print(submit_base['teamid'].drop_duplicates().count())
print(fe_set['teamid'].drop_duplicates().count())
fe_result = submit_base.merge(fe_set, on=['teamid', 'season'], how='inner')
print(fe_result['teamid'].drop_duplicates().count())
fe_result = fe_result.merge(lose_team_fe, on=['teamid_2', 'season'], how='inner')
print(fe_result['teamid'].drop_duplicates().count())

label = LabelEncoder()
fe_result['teamid_label'] = label.fit_transform(fe_result['teamid'])
fe_result['teamid_2_label'] = label.fit_transform(fe_result['teamid_2'])
fe_result['coachname'] = label.fit_transform(fe_result['coachname'])
fe_result['coachname_2'] = label.fit_transform(fe_result['coachname_2'])
#  fe_result['location'] = 0
fe_result['daynum'] = 134

print(fe_result.count())
print(fe_result['teamid'].drop_duplicates().count())
fe_result.to_csv(output_path, index=False)
