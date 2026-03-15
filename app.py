from src.entity import TrainingPipelineConfig
from src.entity.data_ingestion_entity import DataIngestionEntityConfig
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    training_pipleline_config = TrainingPipelineConfig()
    data_ingestion_entity_config = DataIngestionEntityConfig(training_pipeline_config=training_pipleline_config)
    data_ingestion = DataIngestion(data_ingestion_entity_config=data_ingestion_entity_config)
    data_ingestion_artifact_entity = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact_entity)