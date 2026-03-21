import mlflow, dagshub, os
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sagemaker.core.helper.session_helper import Session

from src.utils.utils import save_object, load_numpy_array_data, save_object_as_tarfile
from src.entity.model_trainer_entity import ModelTrainerEntityConfig, ModelTrainerEntityArtifact
from src.entity.data_transformation_entity import DataTransformationArtifactEntity

load_dotenv()

sagemaker_session = Session()

class ModelTrainer:
    def __init__(self, model_trainer_entity_config: ModelTrainerEntityConfig, data_transformation_artifact_entity: DataTransformationArtifactEntity):
        self.model_trainer_entity_config = model_trainer_entity_config
        self.data_transformation_artifact_entity = data_transformation_artifact_entity
    

    
    def initiate_model_trainer(self) -> ModelTrainerEntityArtifact:
        try:
            dagshub.init(repo_name=os.getenv("DAGSHUB_REPO_NAME"), repo_owner=os.getenv("DAGSHUB_REPO_OWNER"), mlflow=True)
            mlflow.set_experiment("Walmart Sales Prediction Project")
            mlflow.end_run()
            train_arr_path = self.data_transformation_artifact_entity.data_transformation_train_file_path
            test_arr_path = self.data_transformation_artifact_entity.data_transformation_test_file_path

            train_arr = load_numpy_array_data(train_arr_path)
            test_arr = load_numpy_array_data(test_arr_path)

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            with mlflow.start_run(run_name="Random Forest Regressor"):
                rf = RandomForestRegressor()
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred_rf)
                r2 = r2_score(y_test, y_pred_rf)
                mse = mean_squared_error(y_test, y_pred_rf)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mse", mse)

            model_file_path = self.model_trainer_entity_config.trained_model_file_path
            compressed_model_file_path = self.model_trainer_entity_config.trained_model_tarfile_path
            save_object(file_path=model_file_path, obj=rf)
            save_object_as_tarfile(file_path=model_file_path, compressed_file_path=compressed_model_file_path)
            sagemaker_session.upload_data(path=compressed_model_file_path, bucket="walmart-prediction-storage")
            return ModelTrainerEntityArtifact(trained_model_file_path=model_file_path)        
        except Exception as e:
            raise e



