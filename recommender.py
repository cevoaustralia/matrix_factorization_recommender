# pylint: disable=C0415,C0103,W0201,W0702,W0718
import os
from metaflow import FlowSpec, step
import random

# import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path=".env")
except Exception:
    print("No dotenv package")


class MatrixFactorizationRecommenderPipeline(FlowSpec):
    """
    MatrixFactorizationRecommenderPipeline is an end-to-end flow for a Recommender
    using Matrix Factorization Algorithm
    """

    def upload_to_s3(self, session, filename, bucket, folder="fake_data"):
        """
        upload_to_s3
        """
        try:
            os.chdir(f"./{folder}")
            session.resource("s3").Bucket(bucket).Object(
                f"{folder}/{filename}"
            ).upload_file(filename)
        except Exception as e_xc:
            print("Failed to upload to s3: " + str(e_xc))

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, fail fast here, now.
        """
        assert os.environ["AWS_ACCESS_KEY_ID"]
        assert os.environ["AWS_SECRET_ACCESS_KEY"]
        assert os.environ["AWS_DEFAULT_REGION"]
        assert os.environ["BUCKET_NAME"]

        self.AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
        self.AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
        self.AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
        self.BUCKET_NAME = os.environ["BUCKET_NAME"]
        self.FAKE_DATA_FOLDER = "fake_data"

        self.next(self.data_generation_users, self.data_generation_items)

    @step
    def data_generation_users(self):
        """
        Users Data Generation
        """
        import generators.UsersGenerator as users
        import boto3

        OUT_USERS_FILENAME = f"./{self.FAKE_DATA_FOLDER}/users.csv"
        IN_USERS_FILENAMES = [
            f"./{self.FAKE_DATA_FOLDER}/users.json.gz",
            f"./{self.FAKE_DATA_FOLDER}/cstore_users.json.gz",
        ]
        usersGenerator = users.UsersGenerator(OUT_USERS_FILENAME, IN_USERS_FILENAMES)
        self.users_df = usersGenerator.generate()
        users_filename = "users.csv"

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(session, users_filename, self.BUCKET_NAME)

        self.next(self.data_generation_interactions)

    @step
    def data_generation_items(self):
        """
        Items Data Generation
        """
        import generators.ItemsGenerator as items
        import boto3

        IN_PRODUCTS_FILENAME = "./generators/products.yaml"

        # This is where stage.sh will pick them up from
        ITEMS_FILENAME = "items.csv"
        OUT_ITEMS_FILENAME = f"./{self.FAKE_DATA_FOLDER}/{ITEMS_FILENAME}"

        itemGenerator = items.ItemsGenerator(
            OUT_ITEMS_FILENAME,
            IN_PRODUCTS_FILENAME,
        )

        self.products_df = itemGenerator.generate()

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(session, ITEMS_FILENAME, self.BUCKET_NAME)

        self.next(self.data_generation_interactions)

    @step
    def data_generation_interactions(self, inputs):
        """
        This is a Join Metaflow step
        Performed after generating users and items data
        Interactions Data Generation
        """
        import generators.InteractionsGenerator as interactions
        import boto3

        self.merge_artifacts(inputs)  # Merge artifacts from previous steps
        INTERACTIONS_FILENAME = "interactions.csv"
        OUT_INTERACTIONS_FILENAME = f"./{self.FAKE_DATA_FOLDER}/{INTERACTIONS_FILENAME}"

        interactionsGenerator = interactions.InteractionsGenerator(
            OUT_INTERACTIONS_FILENAME,
            self.users_df,
            self.products_df,
        )
        interactionsGenerator.generate()

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(session, INTERACTIONS_FILENAME, self.BUCKET_NAME)
        print("Interactions Data Generation...")

        self.next(self.data_transformation)

    @step
    def data_transformation(self):
        """
        Data Transformation
        """
        import pandas as pd
        import boto3

        df = pd.read_csv(f"./{self.FAKE_DATA_FOLDER}/interactions.csv")
        df_items = pd.read_csv(f"./{self.FAKE_DATA_FOLDER}/items.csv")

        # drop columns which we don't need
        df = df.drop(["TIMESTAMP", "DISCOUNT"], axis=1)

        # add confidence scores
        event_type_confidence = {
            "View": 1.0,
            "AddToCart": 2.0,
            "ViewCart": 3.0,
            "StartCheckout": 4.0,
            "Purchase": 5.0,
        }

        # add confidence scores based on the event type defeind above
        df["CONFIDENCE"] = df["EVENT_TYPE"].apply(lambda x: event_type_confidence[x])

        # this removes duplicates and adds up the confidence => down lower number of unique user-item interactions + confidence
        grouped_df = df.groupby(["ITEM_ID", "USER_ID"]).sum("CONFIDENCE").reset_index()
        grouped_df = grouped_df[
            ["USER_ID", "ITEM_ID", "CONFIDENCE"]
        ]  # re-order columns

        # prepare for training
        grouped_df["USER_ID"] = grouped_df["USER_ID"].astype("category")
        grouped_df["ITEM_ID"] = grouped_df["ITEM_ID"].astype("category")
        print(f"Number of unique users: {grouped_df['USER_ID'].nunique()}")
        print(f"Number of unique items: {grouped_df.ITEM_ID.nunique()}")
        grouped_df["USER_IDX"] = grouped_df["USER_ID"].cat.codes
        grouped_df["ITEM_IDX"] = grouped_df["ITEM_ID"].cat.codes
        print(f"Min value user index: {grouped_df['USER_IDX'].min()}")
        print(f"Max value user index: {grouped_df['USER_IDX'].max()}")
        print(f"Min value item index: {grouped_df['ITEM_IDX'].min()}")
        print(f"Max value item index: {grouped_df['ITEM_IDX'].max()}")
        IF_BASEFILENAME = "interactions-confidence.csv"
        INTER_CONFIDENCE_FILENAME = f"./{self.FAKE_DATA_FOLDER}/{IF_BASEFILENAME}"
        grouped_df.to_csv(INTER_CONFIDENCE_FILENAME, index=False)

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(session, IF_BASEFILENAME, self.BUCKET_NAME)
        print("Data Transformation...")

        self.next(self.model_training)

    @step
    def model_training(self):
        """
        Model training
        """
        from scipy import sparse
        import implicit
        import pandas as pd
        import pickle
        from time import gmtime, strftime, sleep
        import boto3

        grouped_df = pd.read_csv(
            f"./{self.FAKE_DATA_FOLDER}/interactions-confidence.csv"
        )

        # create the sparse user-item matrix for the implicit library
        sparse_person_content = sparse.csr_matrix(
            (
                grouped_df["CONFIDENCE"].astype(float),
                (grouped_df["USER_IDX"], grouped_df["ITEM_IDX"]),
            )
        )
        print(sparse_person_content.shape)

        # todo: add hyperparameter tuning for alpha, factors, regularization, iterations
        model = implicit.als.AlternatingLeastSquares(
            factors=20, regularization=0.1, iterations=50
        )
        alpha = 15
        model.fit((sparse_person_content * alpha).astype("double"))
        MODELS_FOLDER = "models"
        MODEL_PKL_FILENAME = (
            f"mf-recommender-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}.pkl"
        )
        pickle.dump(model, open(f"./{MODELS_FOLDER}/{MODEL_PKL_FILENAME}", "wb"))
        # sleep(3)

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(
            session, MODEL_PKL_FILENAME, self.BUCKET_NAME, folder=MODELS_FOLDER
        )
        print("Model Training...")

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
