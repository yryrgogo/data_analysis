import numpy as np
import pandas as pd
import sys


##その時間前後の集計###############################

def shift_values(data, particle, at, value, num):

    df = data[list(particle)+[at, value]].drop_duplicates().sort_values(by=at, ascending=True).reset_index(drop=True)

    for i in range(1, num+1, 1):
        tmp = df[value].shift(i).values
        if np.abs(i)==1:
            result1 = tmp
        else:
            result1 += tmp

    for j in range(-1, -1*num-1, -1):
        tmp = df[value].shift(j).values
        if np.abs(j)==1:
            result2 = tmp
        else:
            result2 += tmp

    row_value = df[value].values
    result = (row_value + result1 + result2)/3
    return result
    df['{}_{}_sur'.format(value, at)] = result
    #  df['{}_{}_shift{}_sum'.format(value, at, str(num))] = result1
    #  df['{}_{}_shift{}_sum'.format(value, at, str(-1*num))] = result2

    return df


def one_particle_COUNTD(df, particle_1, value):

    df = df[[particle_1, value]].drop_duplicates()
    df = df.groupby([particle_1]).size().reset_index().rename(
        columns={0: '1_{}_V_@{}'.format(value, particle_1)})

    return df


def two_particle_COUNTD(df, particle_1_2, value):

    df = df[[particle_1_2[0], particle_1_2[1], value]].drop_duplicates()
    result = df.groupby([particle_1_2[0], particle_1_2[1]]).size().reset_index().rename(
        columns={0: '2_{}_V_@{}{}'.format(value, particle_1_2[0], particle_1_2[1])})

    return result


def three_particle_COUNTD(df, particle_1_2_3, value):

    df = df[[particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], value]].drop_duplicates()
    result = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2]]).size().reset_index().rename(
        columns={0: '3_{}_V_@{}{}{}'.format(value, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2])})

    return result


def four_particle_COUNTD(df, particle_1_2_3_4, value):

    df = df[[particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], value]].drop_duplicates()
    result = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3]]).size().reset_index().rename(
        columns={0: '4_{}_V_@{}{}{}'.format(value, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3])})

    return result


def one_particle_3_value_agg(df, particle_1, value_1, value_2, method):

    #  print(particle_1)

    if value_1 == 0:
        df = df.groupby([particle_1])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})

    else:
        df = df.groupby([particle_1])[value_1, value_2].agg(
            {'{}_{}'.format(particle_1, method): '{}'.format(method)})
        df = df['{}_{}'.format(particle_1, method)]

        if method == 'sum':
            df['1_dp_@{}'.format(particle_1)] = df[value_1].values / df[value_2].values

    # featureの粒度が分かるようにrename
    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '1_{}_{}_@{}'.format(col, method, particle_1)}, inplace=True)

    return df


def two_particle_3_value_agg(df, particle_1_2, value_1, value_2, method):

    #  print(particle_1_2)

    if value_1 == 0:
        df = df.groupby([particle_1_2[0], particle_1_2[1]])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2[0], particle_1_2[1]])[value_1, value_2].agg(
            {'{}_{}_{}'.format(particle_1_2[0], particle_1_2[1], method): '{}'.format(method)})
        df = df['{}_{}_{}'.format(particle_1_2[0], particle_1_2[1], method)]

        if method == 'sum':
            df['2_dp_@{}{}'.format(particle_1_2[0], particle_1_2[1])] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '2_{}_{}_@{}{}'.format(col, method, particle_1_2[0], particle_1_2[1])}, inplace=True)

    return df


def three_particle_3_value_agg(df, particle_1_2_3, value_1, value_2, method):

    #  print(particle_1_2_3)

    if value_1 == 0:
        df = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2]])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2]])[value_1, value_2].agg(
            {'{}_{}_{}_{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], method): '{}'.format(method)})
        df = df['{}_{}_{}_{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], method)]

        if method == 'sum':
            df['3_dp_@{}{}{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2])] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '3_{}_{}_@{}{}{}'.format(col, method, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2])}, inplace=True)

    return df


def four_particle_3_value_agg(df, particle_1_2_3_4, value_1, value_2, method):

    #  print(particle_1_2_3_4)

    if value_1 == 0:
        df = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3]])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3]])[value_1, value_2].agg(
            {'{}_{}_{}_{}_{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], method): '{}'.format(method)})
    # 複数valueをaggに入れるとmulti_indexになるので取り出す

        df = df['{}_{}_{}_{}_{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], method)]

        if method == 'sum':
            df['4_dp_@{}{}{}{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3])] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '4_{}_{}_@{}{}{}{}'.format(col, method, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3])}, inplace=True)

    return df


########### include time category ##############


def one_particle_time_COUNTD(df, particle_1, time, value):

    df = df[[particle_1, time, value]].drop_duplicates()
    df = df.groupby([particle_1, time]).size().reset_index().rename(
        columns={0: '1_{}_V_@{}_{}'.format(value, particle_1, time)})

    if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
        df['1_{}_V_@{}_{}_sur'.format(value, particle_1, time)] = shift_values(df, particle_1, time, '1_{}_V_@{}_{}'.format(value, particle_1, time), 3)
        #  df.drop('2_{}_V_@{}{}_{}'.format(value, particle_1_2[0], particle_1_2[1], time), axis=1, inplace=True)

    return df


def two_particle_time_COUNTD(df, particle_1_2, time, value):

    df = df[[particle_1_2[0], particle_1_2[1], time, value]].drop_duplicates()
    df = df.groupby([particle_1_2[0], particle_1_2[1], time]).size().reset_index().rename(
        columns={0: '2_{}_V_@{}{}_{}'.format(value, particle_1_2[0], particle_1_2[1], time)})

    if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
        df['2_{}_V_@{}{}_{}_sur'.format(value, particle_1_2[0], particle_1_2[1], time)] = shift_values(df, particle_1_2, time, '2_{}_V_@{}{}_{}'.format(value, particle_1_2[0], particle_1_2[1], time), 3)
        #  df.drop('2_{}_V_@{}{}_{}'.format(value, particle_1_2[0], particle_1_2[1], time), axis=1, inplace=True)

    return df


def three_particle_time_COUNTD(df, particle_1_2_3, time, value):

    df = df[[particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time, value]].drop_duplicates()
    df = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time]).size().reset_index().rename(
        columns={0: '3_{}_V_@{}{}{}_{}'.format(value, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time)})

    if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
        df['3_{}_V_@{}{}{}_{}_sur'.format(value, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time)] = shift_values(df, particle_1_2_3, time, '3_{}_V_@{}{}{}_{}'.format(value, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time), 3)
        #  df.drop('3_{}_V_@{}{}{}_{}'.format(value, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time), axis=1, inplace=True)

    return df


def four_particle_time_COUNTD(df, particle_1_2_3_4, time, value):

    df = df[[particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time, value]].drop_duplicates()
    df = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time]).size().reset_index().rename(
        columns={0: '4_{}_V_@{}{}{}{}_{}'.format(value, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time)})

    if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
        df['4_{}_V_@{}{}{}{}_{}_sur'.format(value, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time)] = shift_values(df, particle_1_2_3_4, time, '4_{}_V_@{}{}{}{}_{}'.format(value, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time), 3)
        #  df.drop('4_{}_V_@{}{}{}{}_{}'.format(value, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time), axis=1, inplace=True)

    return df


def one_particle_3_value_time_agg(df, particle_1, time, value_1, value_2, method):

    if value_1 == 0:
        df = df.groupby([particle_1, time])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})

    else:
        df = df.groupby([particle_1, time])[value_1, value_2].agg(
            {'{}_{}_{}'.format(particle_1, time, method): '{}'.format(method)})
        df = df['{}_{}_{}'.format(particle_1, time, method)]

        if method == 'sum':
            df['1_dp_@{}_{}'.format(particle_1, time)] = df[value_1].values / df[value_2].values

# featureの粒度が分かるようにrename
    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '1_{}_{}_@{}_{}'.format(col, method, particle_1, time)}, inplace=True)

# 対象時間前後でも集計してfeatureを生成する
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time:continue
        if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
            df['{}_sur'.format(col)] = shift_values(df, particle_1, time, col, 3)
            #  df.drop(col, axis=1, inplace=True)

    return df


def two_particle_3_value_time_agg(df, particle_1_2, time, value_1, value_2, method):

    #  print(particle_1_2)

    if value_1 == 0:
        df = df.groupby([particle_1_2[0], particle_1_2[1], time])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2[0], particle_1_2[1], time])[value_1, value_2].agg(
            {'{}_{}_{}_{}'.format(particle_1_2[0], particle_1_2[1], time, method): '{}'.format(method)})
        df = df['{}_{}_{}_{}'.format(particle_1_2[0], particle_1_2[1], time, method)]

        if method == 'sum':
            df['2_dp_@{}{}_{}'.format(particle_1_2[0], particle_1_2[1], time)] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '2_{}_{}_@{}{}_{}'.format(col, method, particle_1_2[0], particle_1_2[1], time)}, inplace=True)

    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time:continue
        if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
            df['{}_sur'.format(col)] = shift_values(df, particle_1_2, time, col, 3)
            #  df.drop(col, axis=1, inplace=True)


    return df


def three_particle_3_value_time_agg(df, particle_1_2_3, time, value_1, value_2, method):

    #  print(particle_1_2_3)

    if value_1 == 0:
        df = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time])[value_1, value_2].agg(
            {'{}_{}_{}_{}_{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time, method): '{}'.format(method)})
        df = df['{}_{}_{}_{}_{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time, method)]

        if method == 'sum':
            df['3_dp_@{}{}{}_{}'.format(particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time)] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '3_{}_{}_@{}{}{}_{}'.format(col, method, particle_1_2_3[0], particle_1_2_3[1], particle_1_2_3[2], time)}, inplace=True)

    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time:continue
        if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
            df['{}_sur'.format(col)] = shift_values(df, particle_1_2_3, time, col, 3)
            #  df.drop(col, axis=1, inplace=True)

    return df


def four_particle_3_value_time_agg(df, particle_1_2_3_4, time, value_1, value_2, method):

    #  print(particle_1_2_3_4)

    if value_1 == 0:
        df = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time])[value_2].agg({'{}'.format(value_2): '{}'.format(method)})
    else:
        df = df.groupby([particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time])[value_1, value_2].agg(
            {'{}_{}_{}_{}_{}_{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time, method): '{}'.format(method)})
    # 複数valueをaggに入れるとmulti_indexになるので取り出す
        df = df['{}_{}_{}_{}_{}_{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time, method)]

        if method == 'sum':
            df['4_dp_@{}{}{}{}_{}'.format(particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time)] = df[value_1].values / df[value_2].values

    df = df.reset_index()
    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time or col.count('dp_@') or col.count('it_avg'):continue
        df.rename(columns={col: '4_{}_{}_@{}{}{}{}_{}'.format(col, method, particle_1_2_3_4[0], particle_1_2_3_4[1], particle_1_2_3_4[2], particle_1_2_3_4[3], time)}, inplace=True)

    for col in df.columns:
        if col=='o' or col=='d' or col=='a' or col=='c' or col==time:continue
        if not(time.count('d')) and not(time.count('pts')) and not(time.count('val')) :
            df['{}_sur'.format(col)] = shift_values(df, particle_1_2_3_4, time, col, 3)
            #  df.drop(col, axis=1, inplace=True)

    return df
