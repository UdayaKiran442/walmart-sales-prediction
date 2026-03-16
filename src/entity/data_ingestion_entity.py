import os

from src.entity import TrainingPipelineConfig
from src.constants.constants import DATA_INGESTION_DIR, INGESTION_TRAIN_FILE_NAME, INGESTION_TEST_FILE_NAME, INGESTION_DATA_FILE_NAME

class DataIngestionEntityConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR)
        self.ingestion_dir_train_file_path = os.path.join(self.data_ingestion_dir, INGESTION_TRAIN_FILE_NAME)
        self.ingestion_dir_test_file_path = os.path.join(self.data_ingestion_dir, INGESTION_TEST_FILE_NAME)
        self.ingestion_dir_data_file_path = os.path.join(self.data_ingestion_dir, INGESTION_DATA_FILE_NAME)

class DataIngestionArtifactEntity:
    def __init__(self, train_file_path: str, test_file_path: str):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path