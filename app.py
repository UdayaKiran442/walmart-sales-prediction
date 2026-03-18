from src.entity import TrainingPipelineConfig
from src.entity.data_ingestion_entity import DataIngestionEntityConfig
from src.entity.data_transformation_entity import DataTransformationEntityConfig
from src.entity.model_trainer_entity import ModelTrainerEntityConfig, ModelTrainerEntityArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    training_pipleline_config = TrainingPipelineConfig()
    data_ingestion_entity_config = DataIngestionEntityConfig(training_pipeline_config=training_pipleline_config)
    data_ingestion = DataIngestion(data_ingestion_entity_config=data_ingestion_entity_config)
    data_ingestion_artifact_entity = data_ingestion.initiate_data_ingestion()
    print("Completed data ingestion")
    data_transformation_entity_config = DataTransformationEntityConfig(training_pipeline_config=training_pipleline_config)
    data_transformation = DataTransformation(data_transformation_entity_config, data_ingestion_artifact_entity)
    data_transformation_artifact_entity = data_transformation.initiate_data_transformation()
    print("Completed data transformation")
    model_trainer_entity_config = ModelTrainerEntityConfig(training_pipeline_config=training_pipleline_config)
    model_trainer = ModelTrainer(model_trainer_entity_config, data_transformation_artifact_entity)
    model_trainer_entity_artifact = model_trainer.initiate_model_trainer()
    print("Completed model trainer")
    print(model_trainer_entity_artifact)