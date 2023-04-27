import io
from metaflow import FlowSpec, step
import numpy as np
import os

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path=".env")
except:
    print("No dotenv package")


class MatrixFactorizationRecommenderPipeline(FlowSpec):
    """
    MatrixFactorizationRecommenderPipeline is an end-to-end flow for a Recommender using Matrix Factorization Algorithm
    """

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, fail fast here, now.
        """
        import sagemaker
        import boto3

        assert os.environ["SAGEMAKER_EXECUTION_ROLE"]
        assert os.environ["AWS_ACCESS_KEY_ID"]
        assert os.environ["AWS_SECRET_ACCESS_KEY"]
        assert os.environ["METAFLOW_DATASTORE_SYSROOT_S3"]
        assert os.environ["METAFLOW_DATATOOLS_SYSROOT_S3"]
        assert os.environ["METAFLOW_DEFAULT_DATASTORE"]
        assert os.environ["METAFLOW_DEFAULT_METADATA"]

        session = sagemaker.Session()

        self.region = boto3.Session().region_name

        # S3 bucket where the original mnist data is downloaded and stored.
        self.downloaded_data_bucket = f"sagemaker-sample-files"
        self.downloaded_data_prefix = "datasets/image/MNIST"

        # S3 bucket for saving code and model artifacts.
        # Feel free to specify a different bucket and prefix
        self.bucket = session.default_bucket()
        self.prefix = "sagemaker/DEMO-linear-mnist"

        # Define IAM role
        self.role = os.environ["SAGEMAKER_EXECUTION_ROLE"]

        self.next(self.data_ingestion)

    @step
    def data_ingestion(self):
        """
        Data Ingestion
        """

        self.next(self.data_conversion)

    @step
    def data_conversion(self):
        """
        Data Conversion
        """

        self.next(self.upload_training_data)

    @step
    def upload_training_data(self):
        """
        Upload Training Data
        """

        self.next(self.model_training)

    @step
    def model_training(self):
        """
        Model training
        """

        self.next(self.create_sagemaker_model)

    @step
    def create_sagemaker_model(self):
        """
        Create SageMaker Model
        """

        self.next(self.create_sagemaker_endpoint_configuration)

    @step
    def create_sagemaker_endpoint_configuration(self):
        """
        Create SageMaker Endpoint Configuration
        """

        self.next(self.create_sagemaker_endpoint)

    @step
    def create_sagemaker_endpoint(self):
        """
        Create SageMaker Endpoint
        """

        self.next(self.perform_prediction)

    # Simple function to create a csv from our numpy array
    def np2csv(self, arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=",", fmt="%g")
        return csv.getvalue().decode().rstrip()

    @step
    def perform_prediction(self):
        """
        Placeholder for performing prediction on the SageMaker Endpoint
        """

        self.next(self.delete_sagemaker_endpoint)

    @step
    def delete_sagemaker_endpoint(self):
        """
        Delete SageMaker Endpoint - you don't want that AWS bill, do you?
        - after all that work, delete all to avoid a credit card bill :)
        """

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """


if __name__ == "__main__":
    MatrixFactorizationRecommenderPipeline()
