import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.entity.data_ingestion_entity import DataIngestionEntityConfig, DataIngestionArtifactEntity

class DataIngestion:
    def __init__(self, data_ingestion_entity_config: DataIngestionEntityConfig):
        self.data_ingestion_entity_config = data_ingestion_entity_config
    
    def initiate_data_ingestion(self) -> DataIngestionArtifactEntity:
        data_ingested_dir = os.path.dirname(self.data_ingestion_entity_config.ingestion_dir_train_file_path)
        os.makedirs(data_ingested_dir, exist_ok=True)

        features_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/features.csv')
        stores_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/stores.csv')
        data_df = pd.read_csv('/Users/uday/Desktop/walmart-sales-prediction/data/data.csv')

        merged_data_df = data_df.merge(stores_df, how='left').merge(features_df, how='left')
        train, test = train_test_split(merged_data_df, test_size=0.2, random_state=42)


        train.to_csv(self.data_ingestion_entity_config.ingestion_dir_train_file_path, index=False, header=True)
        test.to_csv(self.data_ingestion_entity_config.ingestion_dir_test_file_path, index=False, header=True)
        merged_data_df.to_csv(self.data_ingestion_entity_config.ingestion_dir_data_file_path, index=False, header=True)

        return DataIngestionArtifactEntity(
            train_file_path=self.data_ingestion_entity_config.ingestion_dir_train_file_path,
            test_file_path=self.data_ingestion_entity_config.ingestion_dir_test_file_path
        )


