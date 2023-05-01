import os
from metaflow import FlowSpec, step

# import numpy as np

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
        # import sagemaker
        # import boto3

        # assert os.environ["SAGEMAKER_EXECUTION_ROLE"]
        # assert os.environ["AWS_ACCESS_KEY_ID"]
        # assert os.environ["AWS_SECRET_ACCESS_KEY"]
        # assert os.environ["METAFLOW_DATASTORE_SYSROOT_S3"]
        # assert os.environ["METAFLOW_DATATOOLS_SYSROOT_S3"]
        # assert os.environ["METAFLOW_DEFAULT_DATASTORE"]
        # assert os.environ["METAFLOW_DEFAULT_METADATA"]

        # session = sagemaker.Session()

        # self.region = boto3.Session().region_name

        # # S3 bucket where the original mnist data is downloaded and stored.
        # self.downloaded_data_bucket = f"sagemaker-sample-files"
        # self.downloaded_data_prefix = "datasets/image/MNIST"

        # # S3 bucket for saving code and model artifacts.
        # # Feel free to specify a different bucket and prefix
        # self.bucket = session.default_bucket()
        # self.prefix = "sagemaker/DEMO-linear-mnist"

        # # Define IAM role
        # self.role = os.environ["SAGEMAKER_EXECUTION_ROLE"]
        assert os.environ["AWS_ACCESS_KEY_ID"]
        assert os.environ["AWS_SECRET_ACCESS_KEY"]
        assert os.environ["AWS_DEFAULT_REGION"]
        assert os.environ["BUCKET_NAME"]

        self.AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
        self.AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
        self.AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
        self.BUCKET_NAME = os.environ["BUCKET_NAME"]

        self.next(self.data_generation_users)

    @step
    def data_generation_users(self):
        """
        Users Data Generation
        """
        import generators.UsersGenerator as users
        import boto3
        from os.path import exists

        usersGenerator = users.UsersGenerator()
        usersGenerator.generate()
        bucket = self.BUCKET_NAME
        users_filename = "users.csv"

        # if exists(users_filename):
        #     raise Exception("Users file does not exist")

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        try:
            os.chdir("./fake_data")
            session.resource("s3").Bucket(bucket).Object(
                f"fake_data/{users_filename}"
            ).upload_file(users_filename)
        except Exception as e:
            print("Failed to upload to s3: " + str(e))

        self.next(self.data_generation_items)

    @step
    def data_generation_items(self):
        """
        Items Data Generation
        """
        import generators.ItemsGenerator as items

        IN_PRODUCTS_FILENAME = "./generators/products.yaml"
        IN_USERS_FILENAMES = [
            "./fake_data/users.json.gz",
            "./fake_data/cstore_users.json.gz",
        ]

        # This is where stage.sh will pick them up from
        out_items_filename = f"./fake_data/items.csv"
        out_users_filename = f"./fake_data/users.csv"

        itemGenerator = items.ItemsGenerator(
            out_users_filename,
            out_items_filename,
            IN_USERS_FILENAMES,
            IN_PRODUCTS_FILENAME,
        )

        self.users_df, self.products_df = itemGenerator.generate()
        self.next(self.data_generation_interactions)

    @step
    def data_generation_interactions(self):
        """
        Interactions Data Generation
        """
        import generators.InteractionsGenerator as interactions

        # This is where stage.sh will pick them up from
        self.out_interactions_filename = f"./fake_data/interactions.csv"

        interactionsGenerator = interactions.InteractionsGenerator(
            self.out_interactions_filename, self.users_df, self.products_df
        )
        interactionsGenerator.generate()
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
