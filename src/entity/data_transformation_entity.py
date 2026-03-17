import os

from src.entity import TrainingPipelineConfig

class DataTransformationEntityConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.data_transformation_train_file_path = os.path.join(self.data_transformation_dir, "transformed_data", "train.npy")
        self.data_transformation_test_file_path = os.path.join(self.data_transformation_dir, "transformed_data", "test.npy")
        self.data_transformation_object_file_path = os.path.join(self.data_transformation_dir, "transformated_object", "preprocessor.pkl")

class DataTransformationArtifactEntity:
    def __init__(self, data_transformation_train_file_path: str, data_transformation_test_file_path: str, data_transformation_object_file_path: str):
        self.data_transformation_train_file_path: str = data_transformation_train_file_path
        self.data_transformation_test_file_path: str = data_transformation_test_file_path
        self.data_transformation_object_file_path: str = data_transformation_object_file_path    