import xlearn as xl
import pandas as pd


def ffm_setting():
    # Training task
    ffm_model = xl.create_ffm()  # Use field-aware factorization machine
    ffm_model.setTrain("./small_train.txt")   # Training data
    ffm_model.setValidate("./small_test.txt")  # Validation data

    # param:
    #  0. binary classification
    #  1. learning rate : 0.2
    #  2. regular lambda : 0.002
    param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002}

    # Train model
    ffm_model.fit(param, "./model.out")


def convert_to_ffm(df, numerics, categories, features, fname):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open(f"../output/{fname}_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['target']))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x] == 0):
                    datastring = datastring + " " + \
                        str(i)+":" + str(i)+":" + str(datarow[x])
                else:
                    # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        # encoding the feature
                        catcodes[x][datarow[x]] = currentcode
            # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode += 1
                        # encoding the feature
                        catcodes[x][datarow[x]] = currentcode
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + \
                        str(i)+":" + str(int(code))+":1"

            datastring += '\n'
            text_file.write(datastring)
