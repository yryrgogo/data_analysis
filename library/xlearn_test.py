import xlearn as xl

# Training task
ffm_model = xl.create_ffm()  # Use field-aware factorization machine
ffm_model.setTrain("./small_train.txt")   # Training data
ffm_model.setValidate("./small_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}

# Train model
ffm_model.fit(param, "./model.out")
