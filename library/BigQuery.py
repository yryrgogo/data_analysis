import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger


def mkdir_func(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def logger_func(OUTPUT_DIR='../output'):
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s]\
    [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    mkdir_func(OUTPUT_DIR)
    handler = FileHandler('{}/py_train.py.log'.format(OUTPUT_DIR), 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    return logger

class BigQuery:

    def __init__(self, credentials, dataset_name, is_create=False, OUTPUT_DIR='../output'):
        self.logger = logger_func(OUTPUT_DIR=OUTPUT_DIR)
        # self.client = bigquery.Client()
        self.client = bigquery.Client.from_service_account_json(credentials)
        self.dataset_name = dataset_name
        if not is_create:
            self._set_dataset()
        self.table_dict = {}

    def _set_dataset(self):
        dataset_ref = self.client.dataset(self.dataset_name)
        self.dataset = self.client.get_dataset(dataset_ref)
        self.logger.info('Setup Dataset {}.'.format(self.dataset.dataset_id))

    def set_table(self, table_name):
        table_ref = self.dataset.table(table_name)
        self.table_dict[table_name] = self.client.get_table(table_ref)
        self.logger.info('Setup Table {}.'.format(self.table_dict[table_name].table_id))

    def create_dataset(self):
        dataset_ref = self.client.dataset(self.dataset_name)
        dataset = bigquery.Dataset(dataset_ref)
        self.dataset = self.client.create_dataset(dataset)

        self.logger.info('Dataset {} created.'.format(self.dataset.dataset_id))

    def create_table(self, table_name, schema):

        table_ref = self.dataset.table(table_name)
        table = bigquery.Table(table_ref, schema=schema)
        self.table_dict[table_name] = self.client.create_table(table)

        self.logger.info('Table {} created.'.format(self.table_dict[table_name].table_id))

    def create_schema(self, column_names, column_types):
        schema = []
        for col_name, col_type in zip(column_names, column_types):
            schema.append(bigquery.SchemaField(col_name, col_type, mode='REQUIRED'))
        return schema

    def insert_rows(self, table_name, insert_rows):
        res = self.client.insert_rows(self.table_dict[table_name], insert_rows)
        if res:
            self.logger.info("Insert Error!!: {}".format(res))

    def del_table(self, table_name):

        dataset_ref = self.client.dataset(self.dataset_name)
        table_ref = self.dataset.table(table_name)
        res = self.client.delete_table(table_ref)
        self.logger.info("del table: {} | Res: {}".format(table_ref, res))

    def del_dataset_all(self):

        dataset_ref = self.client.dataset(self.dataset_name)
        table_ref_list = list(self.client.list_tables(dataset_ref))

        for table_ref in table_ref_list:
            self.client.delete_table(table_ref)
            self.logger.info("del table: {}".format(table_ref))
        self.client.delete_dataset(dataset_ref)
        self.logger.info("del dataset: {}".format(dataset_ref))

    def insert_from_gcs(self, table_name, bucket_name, blob_name, source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1):

        table_ref = self.dataset.table(table_name)

        self.gcs_url = "gs://{}/{}".format(bucket_name, blob_name)

        job_id_prefix = 'go_job'
        job_config = bigquery.LoadJobConfig()
        job_config.skip_leading_rows = skip_leading_rows
        job_config.source_format = source_format

        load_job = self.client.load_table_from_uri(
            self.gcs_url,
            table_ref,
            job_config=job_config,
            job_id_prefix=job_id_prefix
        )

        self.logger.info("Insert to BigQuery from GCS Start! {} ".format(self.gcs_url))
        self.logger.info(load_job.state)
        self.logger.info(load_job.job_type)
        assert load_job.state == 'RUNNING'
        assert load_job.job_type == 'load'

        load_job.result()  # Waits for table load to complete

        self.logger.info(load_job.state)
        self.logger.info(load_job.job_id)
        assert load_job.state == 'DONE'
        assert load_job.job_id.startswith(job_id_prefix)

