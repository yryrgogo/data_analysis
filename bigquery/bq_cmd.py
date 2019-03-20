import os


#========================================================================
# Create New Dataset
new_dataset = ""
bq_cmd = f"bq --location=US mk {new_dataset}"

# Delete Dataset
bq_cmd = f"bq --location=US rm {new_dataset}"

# Delete Table
table = ""
bq_cmd = f"bq --location=US rm {new_dataset}.{table}"
#========================================================================


#========================================================================
# Create New Table from Local Dataset

import os
option = "--autodetect"
dataset = 'hori'
new_table = 'credit'
data_path = '../input/credit_card_balance.csv'
bq_cmd = f"bq load {option} {dataset}.{new_table} {data_path}"
os.system(bq_cmd)
#========================================================================


#========================================================================
# Create New Table from Query

query = """ select * from `hori.installment` limit 10; """
project = 'horikoshi-ml-224313'
dataset = 'hori'
new_table = 'tmp'
bq_cmd = f"bq --location=US query --destination_table {project}:{dataset}.{new_table} --use_legacy_sql=false '{query}'"
os.system(bq_cmd)
#========================================================================
