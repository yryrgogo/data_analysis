#  from model.lightgbm_ex import lightgbm_ex as lgb_ex
from model.class_mp_test import multi_test

c_list = [
    ['aaa', 'ddd', 'fff']
    ,['bbb', 'eee', 'hhh']
    ,['ccc', 'ggg', 'aas']
]

mt = multi_test()
mt.pararell_comment(c_list)
