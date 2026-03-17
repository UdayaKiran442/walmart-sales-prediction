import pandas as pd
import numpy as np

from src.entity.data_ingestion_entity import DataIngestionArtifactEntity
from src.entity.data_transformation_entity import DataTransformationEntityConfig, DataTransformationArtifactEntity

from src.utils.utils import split_date, clean_df, get_transformation_pipeline, save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_transformation_entity_config: DataTransformationEntityConfig, data_ingestion_artifact_entity: DataIngestionArtifactEntity):
        self.data_transformation_entity_config = data_transformation_entity_config
        self.data_ingestion_artifact_entity = data_ingestion_artifact_entity

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact_entity.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact_entity.test_file_path)

            train_df = split_date(train_df)
            test_df = split_date(test_df)

            train_df = clean_df(train_df)
            test_df = clean_df(test_df)
            
            X_train = train_df.drop(['Weekly_Sales'], axis=1)
            y_train = train_df['Weekly_Sales']
            X_test = test_df.drop(['Weekly_Sales'], axis=1)
            y_test = test_df['Weekly_Sales']

            pipeline = get_transformation_pipeline(binary_cols=['IsHoliday'], numeric_cols=[col for col in X_train.columns if col not in ['IsHoliday']])
            transformation_object = pipeline.fit(X_train)
            X_train_transformed = transformation_object.transform(X_train)
            X_test_transformed = transformation_object.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_numpy_array_data(file_path=self.data_transformation_entity_config.data_transformation_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_entity_config.data_transformation_test_file_path, array=test_arr)
            save_object(file_path=self.data_transformation_entity_config.data_transformation_object_file_path, obj=transformation_object)

            return DataTransformationArtifactEntity(
                data_transformation_object_file_path=self.data_transformation_entity_config.data_transformation_object_file_path,
                data_transformation_test_file_path=self.data_transformation_entity_config.data_transformation_test_file_path,
                data_transformation_train_file_path=self.data_transformation_entity_config.data_transformation_train_file_path
            )


        except Exception as e:
            print(f"Error occurred during data transformation: {e}")
            raise e