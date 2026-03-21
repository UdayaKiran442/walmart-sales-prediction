import os

from src.entity import TrainingPipelineConfig
from src.constants.constants import MODEL_TRAINER_DIR, TRAINED_MODEL_FILE_NAME, COMPRESSED_MODEL_DIR, COMPRESSED_MODEL_FILE_NAME

class ModelTrainerEntityConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, TRAINED_MODEL_FILE_NAME)
        self.trained_model_tarfile_path = os.path.join(self.model_trainer_dir, COMPRESSED_MODEL_DIR, COMPRESSED_MODEL_FILE_NAME)


class ModelTrainerEntityArtifact:
    def __init__(self, trained_model_file_path: str):
        self.trained_model_file_path = trained_model_file_path