import sys
import pickle
import datetime, time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

pd.set_option("display.max_columns", 100)

# 回帰用

# time count
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
print(time.ctime(time.time()))

logger = getLogger(__name__)

# path
input_path = "../input/test/tabelog_store_data_saitama.csv"
pred_path  = "../input/tabelog_pred_minato.csv"
origin_path  = "../input/tabelog_store_data_minato.csv"
log_path   = "../log/train.py.log"
output_path   = "../output/"
model_path = "../output/model/"
model_f    = "20180128_003342tb_xgb.pkl"

# hyper parameter tune
all_params = {
    'max_depth':[3,5,7,9],
    'learning_rate':[0.1],
    'min_child_weight':[3,5,10],
    'n_estimators':[1, 10, 100, 1000, 10000],
    'colsample_bytree':[0.8, 0.9],
    'colsample_bylevel':[0.8, 0.9],
    'reg_alpha':[0,0.1],
    'max_delta_step':[0.1],
    'seed':[0]
}


def train_phase():
    
    # target score = mse
    min_score = 1000
    min_params = None
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # パラメータ総当たり
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        reg = xgb.XGBRegressor(**params)

        list_mse = cross_val_score(reg, x_train, y_train, scoring='neg_mean_squared_error', cv=kfold, n_jobs=-1)

        mse = -np.mean(list_mse)

        logger.info('cv mse: {}'.format(mse))
        logger.debug('cv mse: {}'.format(mse))

        if min_score > mse:
            min_score = mse
            min_params = params
        logger.info('current min mse: {}, params: {}'.format(min_score, min_params))
#         break # 動作確認用

    reg = GridSearchCV(
        xgb.XGBRegressor(),
        all_params,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    reg.fit(x_train, y_train)

    min_params = reg.best_params_

    logger.info('min params: {}'.format(min_params))
#     logger.info('min mse: {}'.format(min_score))
    logger.debug('best params: {}'.format(min_params))

    reg = xgb.XGBRegressor(**min_params)
    reg.fit(x_train, y_train)

    pickle.dump(reg, open(model_path + start_time + "tb_xgb.pkl", "wb"))
    reg = pickle.load(open(model_path + start_time + "tb_xgb.pkl", "rb"))
    
    logger.info('train end')
    logger.debug('train end')
    
    return reg


def test_phase(model):
    
    x_test = df_test.drop('rate', axis=1)
    y_test = df_test['rate'].values

    logger.info('test data load end {}'.format(x_test.shape))

    pred_test = model.predict(x_test)

    mse = mean_squared_error(y_test, pred_test)

    logger.info('test mse : {}'.format(mse))
    logger.debug('test mse : {}'.format(mse))

    importances = pd.Series(model.booster().get_score(importance_type='weight'), index = use_cols)
    importances = importances.sort_values(ascending=False)
    logger.info("imporance in the xgboost Model")
    logger.debug("imporance in the xgboost Model")
    logger.info('{}'.format(importances))
    logger.debug('{}'.format(importances))

    logger.info('test end')
    

# 予測と実データの比較
def prediction(model):
    
    pred_data = pd.read_csv(pred_path)
    pred_data = pred_data[pred_data.rate!=0.0]
    x_pred    = pred_data[use_cols]
    y_pred    = pred_data['rate'].values
    
    logger.info('pred data load end {}'.format(pred_data.shape))

    result_pred = model.predict(x_pred)

    mse = mean_squared_error(y_pred, result_pred)

    logger.info('result mse : {}'.format(mse))
    logger.debug('result mse : {}'.format(mse))
    
    origin = pd.read_csv(origin_path)
    origin = origin[origin.seat!='seat']
    rate = origin['rate'].astype('float').values
    origin = origin[origin.rate!=0.0]
    mu = rate.mean()
    se = rate.std()
    origin_rate = (y_pred * se) + mu
    pred_rate = (result_pred * se) + mu

    x_pred['origin']  = origin_rate
    x_pred['predict'] = pred_rate
    x_pred['diff']    = pred_rate - origin_rate

    x_pred.to_csv(output_path + start_time + "tabelog_predict_result.csv")

    
def main():
    
    model = train_phase()
    
#     model = pickle.load(open(model_path + model_f, 'rb'))

    test_phase(model)

    prediction(model)
    

def dataset_split(df):
    N = df.shape[0]
    N_train = int(df.shape[0] * 0.8)
    print('N:{}, N_train:{}, N_test:{}'.format(N, N_train, N-N_train))

    df_sample = df.sample(n=N, random_state=0)
    train = df_sample[:N_train]
    test  = df_sample[N_train:]
    
    print(train.head())
    print(test.head())

    return train, test
    

if __name__ == '__main__':

    # get log
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_path, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    data = pd.read_csv(input_path)
    df_train, df_test = dataset_split(data)

    x_train = df_train.drop('rate', axis=1)
    y_train = df_train['rate'].values
    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(x_train.shape))

    main()
