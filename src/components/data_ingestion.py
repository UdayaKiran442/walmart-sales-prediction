import os
import pandas as pd

from src.entity.data_ingestion_entity import DataIngestionEntityConfig, DataIngestionArtifactEntity

class DataIngestion:
    def __init__(self, data_ingestion_entity_config: DataIngestionEntityConfig):
        self.data_ingestion_entity_config = data_ingestion_entity_config
    
    def initiate_data_ingestion(self) -> DataIngestionArtifactEntity:
        data_ingested_dir = os.path.dirname(self.data_ingestion_entity_config.ingested_dir_train_file_path)
        os.makedirs(data_ingested_dir, exist_ok=True)

        features_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/features.csv')
        stores_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/stores.csv')
        train_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/train.csv')
        test_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/test.csv')

        merged_train_df = train_df.merge(stores_df, how='left').merge(features_df, how='left')
        merged_test_df = test_df.merge(stores_df, how='left').merge(features_df, how='left')

        merged_train_df.to_csv(self.data_ingestion_entity_config.ingested_dir_train_file_path, index=False, header=True)
        merged_test_df.to_csv(self.data_ingestion_entity_config.ingested_dir_test_file_path, index=False, header=True)

        return DataIngestionArtifactEntity(
            train_file_path=self.data_ingestion_entity_config.ingested_dir_train_file_path,
            test_file_path=self.data_ingestion_entity_config.ingested_dir_test_file_path
        )


