import os

from src.entity import TrainingPipelineConfig

from src.constants.constants import DATA_TRANSFORMATION_DIR, TRANSFORMED_DATA_DIR, TRANSFORMED_OBJECT_DIR, TRANSFORMED_TRAIN_FILE_NAME, TRANSFORMED_TEST_FILE_NAME, TRANSFORMED_OBJECT_FILE_NAME, COMPRESSED_OBJECT_DIR, COMPRESSED_OBJECT_FILE_NAME

class DataTransformationEntityConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR)
        self.data_transformation_train_file_path = os.path.join(self.data_transformation_dir, TRANSFORMED_DATA_DIR, TRANSFORMED_TRAIN_FILE_NAME)
        self.data_transformation_test_file_path = os.path.join(self.data_transformation_dir, TRANSFORMED_DATA_DIR, TRANSFORMED_TEST_FILE_NAME)
        self.data_transformation_object_file_path = os.path.join(self.data_transformation_dir, TRANSFORMED_OBJECT_DIR, TRANSFORMED_OBJECT_FILE_NAME)
        self.data_transformation_object_tarfile_path = os.path.join(self.data_transformation_dir, COMPRESSED_OBJECT_DIR, COMPRESSED_OBJECT_FILE_NAME)

class DataTransformationArtifactEntity:
    def __init__(self, data_transformation_train_file_path: str, data_transformation_test_file_path: str, data_transformation_object_file_path: str):
        self.data_transformation_train_file_path: str = data_transformation_train_file_path
        self.data_transformation_test_file_path: str = data_transformation_test_file_path
        self.data_transformation_object_file_path: str = data_transformation_object_file_path    