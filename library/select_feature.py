code = 0
code = 1
#  code=2
import pandas as pd
import os
import shutil
import sys
import glob
import re

unique_id = 'SK_ID_CURR'
p_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_list = [unique_id, target, 'valid_no',
                   'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV']


def move_to_second_valid(best_select=[], rank=0, key_list=[]):

    if len(best_select) == 0:
        #  best_select = pd.read_csv('../output/use_feature/feature869_importance_auc0.806809193200456.csv')
        best_select = pd.read_csv(
            '../output/cv_feature1350_importances_auc_0.809525405635011.csv')
        #  best_select = pd.read_csv('../output/cv_feature1330_importances_auc_0.8066523340763816.csv')
        #  best_select = pd.read_csv('../output/cv_feature1099_importances_auc_0.8072030486159842.csv')
        best_feature = best_select['feature'].values
        #  best_feature = best_select.query("flg==1")['feature'].values
        #  best_feature = best_select.query("rank<=50")['feature'].values
        #  best_feature = best_select.query("rank>=750")['feature'].values
        #  best_feature = best_select.query("rank>=100")['feature'].values
        #  best_feature = best_select.query("rank>=1000")['feature'].values
        #  best_feature = best_select.query("rank>=1300")['feature'].values
        best_feature = best_select.query("rank>=1200")['feature'].values
        #  best_feature = [col for col in best_feature if (col.count('a_') and col.count('AMT')) or col.count('p_Ap') or col.count('is_rm')]
        #  best_feature = [col for col in best_feature if col.count('impute')]
        #  best_feature = [col for col in best_feature if col.count('ker_')]
        best_feature = [col for col in best_feature if col.count(
            'dima_') or col.count('gp_')]
        #  best_feature = [col for col in best_feature if col.count('new_len')]
        #  best_feature = [col for col in best_feature if col.count('dima_')]
        #  best_feature = [col for col in best_feature if col.count('gp_')]
        #  best_feature = [col for col in best_feature if col.count('new_len')]

        if len(best_feature) == 0:
            sys.exit()
        for feature in best_feature:
            if feature not in ignore_list:
                try:
                    shutil.move(
                        f'../features/3_winner/{feature}.npy', '../features/1_third_valid/')
                    #  shutil.move(f'../features/3_winner/{feature}.npy', '../features/1_second_valid/')
                    #  shutil.move(f'../features/feat_high_cv_overfit/{feature}.npy', '../features/1_third_valid/')
                except FileNotFoundError:
                    pass
        print(f'move to third_valid:{len(best_feature)}')

    else:
        tmp = best_select.query(f"rank>={rank}")['feature'].values
        for key in key_list:
            best_feature = [col for col in tmp if col.count(key)]

            if len(best_feature) == 0:
                sys.exit()
            for feature in best_feature:
                if feature not in ignore_list:
                    shutil.move(
                        f'../features/3_winner/{feature}.npy', '../features/1_third_valid')
            print(f'move to third_valid:{len(best_feature)}')


def move_to_use():

    #  best_select = pd.read_csv('../output/cv_feature1476_importances_auc_0.8091815613330919.csv')
    best_select = pd.read_csv(
        '../output/cv_feature1350_importances_auc_0.809525405635011.csv')
    #  best_select = pd.read_csv('../output/cv_feature1234_importances_auc_0.8091839448990605.csv')
    #  best_select = pd.read_csv('../output/cv_feature1194_importances_auc_0.809452251037472.csv')
    best_feature = best_select['feature'].values
    #  best_feature = best_select.query('flg_2==0')['feature'].values
    #  best_feature = best_select.query('flg==0')['feature'].values

    #  path_list_imp = glob.glob('../features/3_winner/*.npy')
    #  impute_list = []
    #  for path in path_list_imp:
    #      imp_name = re.search(r'/([^/.]*).npy', path).group(1)[:-7]
    #      impute_list.append(imp_name)
    #  best_feature = dima_list

    #  path_list = glob.glob('../features/1_second_valid/*.npy')
    path_list = glob.glob('../features/1_third_valid/*.npy')
    #  path_list = glob.glob('../features/win_tmp/*.npy')
    #  path_list = glob.glob('../features/dima/*.npy')

    for path in path_list:
        filename = re.search(r'/([^/.]*).npy', path).group(1)
        #  if filename in impute_list:
        #      print(f'continue: {filename}')
        #      continue
        #  if filename.count('NAME') or filename.count('TYPE') or filename.count('GENDER'):
        if filename in best_feature:
            #  shutil.move(path, '../features/1_third_valid/')
            shutil.move(path, '../features/3_winner/')
            #  shutil.move(path, '../features/CV08028/')
            #  shutil.move(path, '../features/feat_high_cv_overfit')
        #  for dima in dima_list:
        #      if filename.count(dima):
        #          shutil.move(path, '../features/dima_tmp/')


def move_diff_file(path_1, path_2, move_path):
    '''
    path_1とpath_2にあるファイル名を比較し、
    path_2のファイル名を含むpath_1のファイルをmove_pathへ移す
    '''

    path_list1 = glob.glob(path_1)
    path_list2 = glob.glob(path_2)

    for p_1 in path_list1:
        filename_1 = re.search(r'/([^/.]*).npy', p_1).group(1)
        duplicate_flg = 0
        for p_2 in path_list2:
            filename_2 = re.search(r'/([^/.]*).npy', p_2).group(1)
            print(filename_1)
            print(filename_2)
            if filename_1.count(filename_2):
                duplicate_flg = 1
        if duplicate_flg == 0:
            shutil.move(p_1, move_path)


def move_feature(feature_name, feature_path='../features/4_winner/*.fp', move_path='../features/9_delete'):

    path_list = glob.glob(feature_path)

    for path in path_list:
        if path.count(feature_name):
            shutil.move(path, move_path)


def main():
    if code == 0:
        move_to_second_valid()
    elif code == 1:
        move_to_use()
    elif code == 2:
        move_file()


if __name__ == '__main__':

    main()
