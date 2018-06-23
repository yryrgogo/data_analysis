import pandas as pd


""" 日時操作系 """
def date_diff(start, end):
    diff = end - start
    return diff


def date_range(data, start, end, include_flg=1):
    '''
    include_flgが0の場合, endの日付は含めずにデータを返す
    '''
    if include_flg == 0:
        return data[(start <= data['visit_date']) & (data['visit_date'] < end)]
    return data[(start <= data['visit_date']) & (data['visit_date'] <= end)]


