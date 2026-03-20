import mlflow, dagshub, os
from dotenv import load_dotenv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sagemaker.core.helper.session_helper import Session

from src.utils.utils import save_object, load_numpy_array_data
from src.entity.model_trainer_entity import ModelTrainerEntityConfig, ModelTrainerEntityArtifact
from src.entity.data_transformation_entity import DataTransformationArtifactEntity

load_dotenv()

sagemaker_session = Session()

class ModelTrainer:
    def __init__(self, model_trainer_entity_config: ModelTrainerEntityConfig, data_transformation_artifact_entity: DataTransformationArtifactEntity):
        self.model_trainer_entity_config = model_trainer_entity_config
        self.data_transformation_artifact_entity = data_transformation_artifact_entity
    

    
    def initiate_model_trainer(self) -> ModelTrainerEntityArtifact:
        dagshub.init(repo_name=os.getenv("DAGSHUB_REPO_NAME"), repo_owner=os.getenv("DAGSHUB_REPO_OWNER"), mlflow=True)
        mlflow.set_experiment("Walmart Sales Prediction Experiment")
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
            param_grid = {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 30}
            rf = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_distributions=param_grid,
                cv=3,
                n_iter=10
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_rf)
            r2 = r2_score(y_test, y_pred_rf)
            mse = mean_squared_error(y_test, y_pred_rf)
            mlflow.log_params(rf.best_params_)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)
            print("Logging Random Forest Regressor model to MLflow...")
            mlflow.sklearn.log_model(rf, "model")
            print("Random Forest Regressor model logged to MLflow.")
        mlflow.end_run()
        

        # choose best model from mlflow and save it to disk
        print("Selecting best model based on R2 score...")
        df = mlflow.search_runs(filter_string='metrics.r2 > 0.9')
        if df.empty:
            print("No runs found with R2 score greater than 0.9. Selecting best model from all runs.")
            raise Exception("No runs found with R2 score greater than 0.9. Please check your models and try again.")
        print(df)
        best_run_id = df.loc[df['metrics.r2'].idxmax()]['run_id']
        print("Best Run ID:", best_run_id)
        model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
        print("Best Model:", model)
        model_file_path = self.model_trainer_entity_config.trained_model_file_path
        print("Saving model to:", model_file_path)
        save_object(file_path=model_file_path, obj=model)
        # compress the model file and upload to s3 using tarfile and sagemaker session
        sagemaker_session.upload_data(path=model_file_path, bucket="<write bucket name here>")
        return ModelTrainerEntityArtifact(trained_model_file_path=model_file_path)
       
        
            



