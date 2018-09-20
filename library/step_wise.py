def incremental_decrease(base, path, decrease_path, decrease_word='', dummie=0, val_col='valid_no_4', iter_no=20):

    dataset = make_feature_set(base, path)
    dataset = dataset.set_index(unique_id)

    train = dataset.query('is_train==1')
    logger.info(f'train shape: {train.shape}')
    train.drop(['is_train', 'is_test'], axis=1, inplace=True)

    decrease_list = []

    df_feature = pd.read_csv(decrease_path)
    #  feature_arr_0 = df_feature.query('rank>100').query('rank<=150')['feature'].values
    #  feature_arr_1 = df_feature.query('rank>150').query('rank<=200')['feature'].values
    #  feature_arr_2 = df_feature.query('rank>200').query('rank<=300')['feature'].values
    #  feature_arr_3 = df_feature.query('rank>200')['feature'].values
    feature_arr_9 = df_feature.query('rank>50')['feature'].values
    feature_arr_9 = [col for col in feature_arr_9 if col.count('TARGET')]

    best_score = 0
    score_list = []
    decrease_set_list = []
    iter_no = iter_no
    raw_cols = list(train.columns)

    for i in range(iter_no):

        np.random.seed(np.random.randint(100000))
        #  decrease_list_0 = list(np.random.choice(feature_arr_3, 2))
        #  decrease_list = list(np.random.choice(feature_arr_9, 3)) + decrease_list_0
        decrease_list = list(np.random.choice(feature_arr_9, 4))

        #  target_flg = 0
        #  while target_flg == 0 and is_flg == 0:
        #  while target_flg <15:
        #      target_flg = 0
        #      np.random.seed(np.random.randint(100000))
            #  decrease_list_0 = list(np.random.choice(feature_arr_0, 1))
            #  decrease_list_0 = []
            #  decrease_list_1 = list(np.random.choice(feature_arr_1, 1))
            #  decrease_list_2 = list(np.random.choice(feature_arr_2, 2))
            #  decrease_list_3 = list(np.random.choice(feature_arr_2, 2))
            #  decrease_list_3 = []
            #  decrease_list = decrease_list_0 + decrease_list_1 + decrease_list_2 + decrease_list_3
            #  decrease_list = list(np.random.choice(feature_arr_9, 15))
            #  index_list = np.random.randint(0, len(feature_arr), size=12)

            #  for decrease in decrease_list:
            #      if decrease.count('TARGET'):
            #          target_flg += 1
                #  else:
                #      target_flg = 0
                #  if decrease.count('TARGET'):
                #      target_flg=1

        if len(decrease_word)>0:
            ' decrease_wordで指定した文字を含むfeatureのみ残す '
            decrease_list = [col for col in decrease_list if col.count(decrease_word)]

        use_cols = raw_cols.copy()

        error_list = []
        for decrease in decrease_list:
            try:
                use_cols.remove(decrease)
            except ValueError:
                error_list.append(decrease)
            logger.info(f'decrease feature: {decrease}')

        logger.info(f'\n** LIST REMOVE ERROR FEATURE: {error_list} **')

        #  tmp_result, col_length = get_cv_result(train=train[use_cols],
        #                             target=target,
        #                             val_col=val_col,
        #                             logger=logger,
        #                             params=train_params
        #                             )

        if len(tmp_result)<=1:
            logger.info(f'\nLOW SCORE is truncate.')
            continue

        ' 追加したfeatureが分かるように文字列で入れとく '
        sc_score = tmp_result['cv_score'].values[0]
        score_list.append(sc_score)
        decrease_set_list.append(str(decrease_list))

        if sc_score > best_score:
            best_score = sc_score
            logger.info(f'\ndecrease: {str(decrease_list)}')
            logger.info(f'\nBest Score Update!!!!!')

            tmp_result['remove_feature'] = str(decrease_list)
            tmp_result.to_csv(f"../output/use_feature/feature{len(use_cols)}_rate{train_params['learning_rate']}_auc{sc_score}.csv", index=False)
        else:
            logger.info(f'\ndecrease: {decrease_list} \nNO UPDATE AUC: {sc_score}')

        logger.info(f'\n\n***** CURRENT BEST_SCORE/ AUC: {best_score} *****')

        if (i+1)%10 == 0 :
            result = pd.Series(data=score_list, index=decrease_set_list, name='score')
            result.sort_values(ascending=False, inplace=True)
            logger.info(f"\n*******Now Feature validation Result*******\n{result.head(10)}\n**************")
            result.to_csv(f'../output/{start_time[:12]}_decrease_feature_validation.csv')

        elif i+1==iter_no:
            result = pd.Series(data=score_list, index=decrease_set_list, name='score')
            result.sort_values(ascending=False, inplace=True)
            logger.info(f"\n*******Feature validation Result*******\n{result.head(20)}\n**************")


def incremental_increase(base, path, input_path, move_path, dummie=0, val_col='valid_no'):

    best_score, train, importance = first_train(base, path, dummie=0, val_col=val_col)

    ' 追加していくfeature '
    feature_path = glob.glob(input_path)
    ' 各学習の結果を格納するDF '
    result = pd.DataFrame([])
    df_idx = base['is_train']
    del base

    for number, path in enumerate(feature_path):

        ' npyを読み込み、file名をカラム名とする'
        feature_name = re.search(r'/([^/.]*).npy', path).group(1)
        feature = pd.Series(np.load(path), name=feature_name)

        ' 結合し学習データのみ取り出す '
        dataset = pd.concat([df_idx, feature], axis=1)
        train[feature_name] = dataset.query('is_train==1')[feature_name]
        del feature
        del dataset
        gc.collect()

        logger.info(f'\niteration no: {number}\nvalid feature: {feature_name}')

        #  tmp_result, col_length = get_cv_result(train=train, target=target, val_col=val_col, logger=logger)

        ' 追加したfeatureが分かるように文字列で入れとく '
        tmp_result['add_feature'] = feature_name

        sc_score = tmp_result['cv_score'].values[0]

        ' 前回のiterationよりスコアが落ちたら、追加した特徴量を落とす '
        if metric == 'auc':
            if sc_score <= best_score:
                train.drop(feature_name, axis=1, inplace=True)
                logger.info(f'\nExtract Feature: {feature_name}')
                shutil.move(path, move_path)

        if metric == 'auc':
            if best_score < sc_score:
                best_score = sc_score
                logger.info(f'\nBest Score Update!!!!!')
                logger.info(f'\nAdd Feature: {feature_name}')
                shutil.move(path, '../features/3_winner/')
                tmp_result.to_csv( f'../output/use_feature/feature_importance_auc{sc_score}.csv', index=False)

        elif metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score
        logger.info(f'\nCurrent best_score: {best_score}')

