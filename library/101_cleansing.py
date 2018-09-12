import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
#  sys.path.append(f"{HOME}/kaggle/github/library/")
sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from logger import logger_func
logger = logger_func()
import eda
from utils import get_categorical_features


#==============================================================================
# pickleにする 
#==============================================================================

def make_pkl():
    pri = pd.read_csv('../input/order_products__prior.csv')
    utils.to_df_pickle(df=pri, path='../input', fname='prior')

    tra = pd.read_csv('../input/order_products__train.csv')
    utils.to_df_pickle(df=tra, path='../input', fname='train')

    order = pd.read_csv('../input/orders.csv')
    utils.to_df_pickle(df=order, path='../input', fname='orders')

#  make_pkl()
#  sys.exit()

utils.start(sys.argv[0])

def clean_app(app):
    logger.info(f'''
    #==============================================================================
    # APPLICATION CLEANSING
    #==============================================================================''')

    revo = 'Revolving loans'
    drop_list = [col for col in app.columns if col.count('is_train') or col.count('is_test') or col.count('valid_no')]
    app.drop(drop_list, axis=1, inplace=True)

    app['AMT_INCOME_TOTAL'] = app['AMT_INCOME_TOTAL'].where(app['AMT_INCOME_TOTAL']<1000000, 1000000)
    app['CODE_GENDER'].replace('XNA', 'F', inplace=True)

    cat_cols = get_categorical_features(data=app, ignore=[])
    for col in cat_cols:
        app[col].fillna('XNA', inplace=True)

    ' revo '
    amt_list = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']
    for col in amt_list:
        app[f'revo_{col}'] = app[col].where(app[f'NAME_CONTRACT_TYPE']==revo, np.nan)

    utils.to_df_pickle(df=app, path='../input', fname='clean_application_train_test')


def clean_bureau():
    logger.info(f'''
    #==============================================================================
    # BUREAU CLEANSING
    #==============================================================================''')

    bur = utils.read_df_pickle(path='../input/bureau*.p')
    bur['DAYS_CREDIT_ENDDATE'] = bur['DAYS_CREDIT_ENDDATE'].where(bur['DAYS_CREDIT_ENDDATE']>-36000, np.nan)
    bur['DAYS_ENDDATE_FACT'] = bur['DAYS_ENDDATE_FACT'].where(bur['DAYS_ENDDATE_FACT']>-36000, np.nan)
    bur['DAYS_CREDIT_UPDATE'] = bur['DAYS_CREDIT_UPDATE'].where(bur['DAYS_CREDIT_UPDATE']>-36000, np.nan)
    bur = utils.to_df_pickle(df=bur, path='../input', fname='clean_bureau')


def clean_prev(pre):
    logger.info(f'''
    #==============================================================================
    # PREV CLEANSING
    #==============================================================================''')

    cash = 'Cash loans'
    revo = 'Revolving loans'
    ' RevolvingではCNT_PAYMENT, AMT系をNULLにする '
    pre = utils.read_df_pickle(path='../input/previous*.p')
    pre['AMT_CREDIT'] = pre['AMT_CREDIT'].where(pre['AMT_CREDIT']>0, np.nan)
    pre['AMT_ANNUITY'] = pre['AMT_ANNUITY'].where(pre['AMT_ANNUITY']>0, np.nan)
    pre['AMT_APPLICATION'] = pre['AMT_APPLICATION'].where(pre['AMT_APPLICATION']>0, np.nan)
    pre['CNT_PAYMENT'] = pre['CNT_PAYMENT'].where(pre['CNT_PAYMENT']>0, np.nan)
    pre['AMT_DOWN_PAYMENT'] = pre['AMT_DOWN_PAYMENT'].where(pre['AMT_DOWN_PAYMENT']>0, np.nan)
    pre['RATE_DOWN_PAYMENT'] = pre['RATE_DOWN_PAYMENT'].where(pre['RATE_DOWN_PAYMENT']>0, np.nan)

    pre['DAYS_FIRST_DRAWING']        = pre['DAYS_FIRST_DRAWING'].where(pre['DAYS_FIRST_DRAWING'] <100000, np.nan)
    pre['DAYS_FIRST_DUE']            = pre['DAYS_FIRST_DUE'].where(pre['DAYS_FIRST_DUE']         <100000, np.nan)
    pre['DAYS_LAST_DUE_1ST_VERSION'] = pre['DAYS_LAST_DUE_1ST_VERSION'].where(pre['DAYS_LAST_DUE_1ST_VERSION'] <100000, np.nan)
    pre['DAYS_LAST_DUE']             = pre['DAYS_LAST_DUE'].where(pre['DAYS_LAST_DUE']           <100000, np.nan)
    pre['DAYS_TERMINATION']          = pre['DAYS_TERMINATION'].where(pre['DAYS_TERMINATION']     <100000, np.nan)
    pre['SELLERPLACE_AREA']          = pre['SELLERPLACE_AREA'].where(pre['SELLERPLACE_AREA']     <200, 200)

    ignore_list = ['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS']
    ' revo '
    for col in pre.columns:
        if col in ignore_list:
            logger.info(f'CONTINUE: {col}')
            continue
        pre[f'revo_{col}'] = pre[col].where(pre[f'NAME_CONTRACT_TYPE']==revo, np.nan)

    pre['NAME_TYPE_SUITE'].fillna('XNA', inplace=True)
    pre['PRODUCT_COMBINATION'].fillna('XNA', inplace=True)

    pre = utils.to_df_pickle(df=pre, path='../input', fname='clean_prev')


def clean_pos(df):
    logger.info(f'''
    #==============================================================================
    # PREV CLEANSING
    #==============================================================================''')

    df = df.query("NAME_CONTRACT_STATUS!='Signed' and NAME_CONTRACT_STATUS!='Approved' and NAME_CONTRACT_STATUS!='XNA'")
    df.loc[(df.NAME_CONTRACT_STATUS=='Completed') & (df.CNT_INSTALMENT_FUTURE!=0), 'NAME_CONTRACT_STATUS'] = 'Active'

    df_0 = df.query('CNT_INSTALMENT_FUTURE==0')
    df_1 = df.query('CNT_INSTALMENT_FUTURE>0')
    df_0['NAME_CONTRACT_STATUS'] = 'Completed'
    df_0.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, False], inplace=True)
    df_0.drop_duplicates('SK_ID_PREV', keep='last', inplace=True)
    df = pd.concat([df_0, df_1], ignore_index=True)
    del df_0, df_1
    gc.collect()

    utils.to_df_pickle(df=df, path='../input', fname='clean_pos')


def clean_ins(df):

    df = df.query("AMT_INSTALMENT>0")

    utils.to_df_pickle(df=df, path='../input', fname='clean_install')


def clean_ccb(df):

    amt_cols = [col for col in df.columns if col.count('AMT')]
    cnt_cols = [col for col in df.columns if col.count('CNT')]
    amt_cnt_cols = list(set(amt_cols+cnt_cols))
    for col in amt_cnt_cols:
        df[col].fillna(0, inplace=True)

    utils.to_df_pickle(df=df, path='../input', fname='clean_ccb')


logger.info(f'''
#==============================================================================
# DATA LOAD
#==============================================================================''')
pri = utils.read_df_pickle(path='../input/prior*.p')
tra = utils.read_df_pickle(path='../input/train*.p')
order = utils.read_df_pickle(path='../input/order*.p')

logger.info(f'''
#==============================================================================
# MAKE EDA TABLE
#==============================================================================''')
pri_eda = eda.df_info(pri)
pri_eda.to_csv('../eda/prior_eda.csv')
tra_eda = eda.df_info(tra)
tra_eda.to_csv('../eda/train_eda.csv')
order_eda = eda.df_info(order)
order_eda.to_csv('../eda/orders_eda.csv')
sys.exit()

clean_app(app)
clean_prev(pre)
#  clean_pos(pos)
#  clean_ins(ins)
#  clean_ccb(ccb)

utils.end(sys.argv[0])

#  pre_eda = eda.df_info(pre)
