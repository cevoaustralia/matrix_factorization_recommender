# pylint: disable=C0415,C0103,W0201,W0702,W0718
import os
from metaflow import FlowSpec, step, Parameter
import random
import numpy as np
import json

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

    def make_train_test_split(self, sparse_matrix, pct_test=0.2):
        """
        Create the train-test split in preparation for model training and evaluation
            - sparse_matrix: our original sparse user-item matrix
            - pct_test: the percentage of randomly chosen user-item interactions to mask in the training set
                this defaults to 20 percent of the sparse matrix
        """
        test_set = (
            sparse_matrix.copy()
        )  # Make a copy of the original set to be the test set.
        training_set = (
            sparse_matrix.copy()
        )  # Make a copy of the original data we can alter as our training set.
        nonzero_inds = (
            training_set.nonzero()
        )  # Find the indices in the ratings data where an interaction exists
        nonzero_pairs = list(
            zip(nonzero_inds[0], nonzero_inds[1])
        )  # Zip these pairs together of user,item index into list
        random.seed(42)  # Set the random seed to zero for reproducibility
        num_samples = int(
            np.ceil(pct_test * len(nonzero_pairs))
        )  # Round the number of samples needed to the nearest integer
        # print(f"Number of samples: {num_samples}")
        samples = random.sample(
            nonzero_pairs, num_samples
        )  # Sample a random number of user-item pairs without replacement
        # print(f"Length nonzero pairs: {len(nonzero_pairs)}")
        # print(f"Length samples: {len(samples)}")
        user_inds = [index[0] for index in samples]  # Get the user row indices
        item_inds = [index[1] for index in samples]  # Get the item column indices
        # print(f"Length user index samples: {len(user_inds)}")
        # print(f"Length item index samples: {len(item_inds)}")
        training_set[
            user_inds, item_inds
        ] = 0  # Assign all of the randomly chosen user-item pairs to zero
        training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
        return (
            training_set,
            test_set,
            user_inds,
        )  # Output the unique list of user indices which were altered

    def average_precision_at_k(self, y_true, y_pred, k_max=0):
        """
        Average Precision at k calculation
        """
        # Check if all elements in lists are unique
        if len(set(y_true)) != len(y_true):
            raise ValueError("Values in y_true are not unique")

        if len(set(y_pred)) != len(y_pred):
            raise ValueError("Values in y_pred are not unique")

        if k_max != 0:
            y_pred = y_pred[:k_max]

        correct_predictions = 0
        running_sum = 0

        for i, yp_item in enumerate(y_pred):
            k = i + 1  # our rank starts at 1

            if yp_item in y_true:
                correct_predictions += 1
                running_sum += correct_predictions / k

        return round(running_sum / len(y_true), 5)

    def call_recommend(self, model, user_items, user_idx, N=20):
        """
        Call the recommend function to get the top N recommendations for a user
        """
        num_recomm = N
        recommendations_raw = model.recommend(
            user_idx, user_items[user_idx], N=num_recomm
        )
        predictions = recommendations_raw[0][
            :num_recomm
        ]  # these are the top n preditictions using the train set

        return predictions

    def get_popular_items(self, N=20):
        """
        Get popular items (to use as baseline)
        """
        popular_items = self.grouped_df.ITEM_ID.value_counts(sort=True).keys()[:N]
        top_N_popular_items = []
        for item in popular_items:
            item_desc = self.df_items.PRODUCT_DESCRIPTION.loc[
                self.df_items.ITEM_ID == item
            ].iloc[0]
            item_index = self.grouped_df.ITEM_IDX.loc[
                self.grouped_df.ITEM_ID == item
            ].iloc[0]
            # print(f"Item ID: {item}, Desc: {item_desc}")
            top_N_popular_items.append(item_index)
        return top_N_popular_items

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
        self.USER_COUNT = 1000  # change to required, 10000 is recommended
        self.INTERACTION_COUNT = (
            50000  # change to required count, 650000 is recommended
        )

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
        usersGenerator = users.UsersGenerator(
            OUT_USERS_FILENAME, IN_USERS_FILENAMES, self.USER_COUNT
        )
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
            self.INTERACTION_COUNT,
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
        import itertools

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
        self.grouped_df = grouped_df
        self.df_items = df_items

        session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(session, IF_BASEFILENAME, self.BUCKET_NAME)
        print("Data Transformation...")

        # sets of hyperparameters to try
        alphas = [50, 60]
        factors = [100, 200]
        regularizations = [0.01, 0.1]
        iterations = [100, 150]
        grid_search = []
        for params in itertools.product(alphas, factors, regularizations, iterations):
            grid_search.append(
                {
                    "ALPHA": params[0],
                    "FACTOR": params[1],
                    "REGULARIZATION": params[2],
                    "ITERATIONS": params[3],
                }
            )
        # we serialize hypers to a string and pass them to the foreach below
        self.hypers_sets = [json.dumps(_) for _ in grid_search]
        self.next(self.model_training, foreach="hypers_sets")

    @step
    def model_training(self):
        """
        Model training
        """
        from scipy import sparse
        import implicit

        # this is the CURRENT hyper param JSON in the fan-out
        # each copy of this step in the parallelization will have its own value
        self.hyper_string = self.input
        self.hypers = json.loads(self.hyper_string)

        # create the sparse user-item matrix for the implicit library
        sparse_person_content = sparse.csr_matrix(
            (
                self.grouped_df["CONFIDENCE"].astype(float),
                (self.grouped_df["USER_IDX"], self.grouped_df["ITEM_IDX"]),
            )
        )

        (
            self.training_set,
            self.test_set,
            self.product_users_altered,
        ) = self.make_train_test_split(sparse_person_content)

        # todo: add hyperparameter tuning for alpha, factors, regularization, iterations
        alpha = self.hypers["ALPHA"]
        factors = self.hypers["FACTOR"]
        regularization = self.hypers["REGULARIZATION"]
        iterations = self.hypers["ITERATIONS"]

        # do the training set first (with the masked items)
        training_model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        training_model.fit(
            (self.training_set * alpha).astype("double"), show_progress=False
        )

        # then do the test set (without the masked items)
        testing_model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        testing_model.fit((self.test_set * alpha).astype("double"), show_progress=False)

        user_indices = self.product_users_altered[:20]
        top_n_popular_items = self.get_popular_items()
        precision_records = []
        precision_records_popular = []

        print(f"Model training hypers: {self.hypers}")
        for user_index in user_indices:
            train_set_predictions = self.call_recommend(
                training_model, self.training_set, user_index
            )
            test_set_predictions = self.call_recommend(
                testing_model, self.test_set, user_index
            )

            precision_records.append(
                self.average_precision_at_k(test_set_predictions, train_set_predictions)
            )

            precision_records_popular.append(
                self.average_precision_at_k(test_set_predictions, top_n_popular_items)
            )

        print(
            f"For top K({len(user_indices)}) recommendations, the average precision (this recsys)                : {np.average(precision_records)}"
        )
        print(
            f"For top K({len(user_indices)}) recommendations, the average precision (popular items recsys)       : {np.average(precision_records_popular)}"
        )
        self.metrics = np.average(precision_records)
        self.training_models = training_model
        print("Model Training...")
        self.next(self.join_runs)

    @step
    def join_runs(self, inputs):
        """
        Join the parallel runs
        """
        import pickle
        from time import gmtime, strftime
        import boto3

        print("Joining the parallel runs...")
        self.results_from_runs = {
            inp.hyper_string: [inp.metrics, inp.training_models] for inp in inputs
        }
        self.best_hypers = sorted(
            self.results_from_runs.items(), key=lambda x: x[1], reverse=True
        )[0]

        METRIC_IDX = 0
        MODEL_IDX = 1
        KEY_IDX = 0
        VALUES_IDX = 1
        self.best_selected_hypers = self.best_hypers[KEY_IDX]
        self.best_selected_model = self.best_hypers[VALUES_IDX][MODEL_IDX]
        self.best_selected_metric = self.best_hypers[VALUES_IDX][METRIC_IDX]
        print(f"Best hyperparameters are: {self.best_selected_hypers}")
        print(f"\n\n====> Best metric is: {self.best_selected_metric}\n\n")

        MODELS_FOLDER = "models"
        MODEL_PKL_FILENAME = (
            f"mf-recommender-{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}.pkl"
        )
        # only save the training model, don't bother with the testing model as that is just for evaluation
        pickle.dump(
            self.best_selected_model,
            open(f"./{MODELS_FOLDER}/{MODEL_PKL_FILENAME}", "wb"),
        )

        session = boto3.Session(
            aws_access_key_id=inputs[0].AWS_ACCESS_KEY_ID,
            aws_secret_access_key=inputs[0].AWS_SECRET_ACCESS_KEY,
            region_name=inputs[0].AWS_DEFAULT_REGION,
        )

        self.upload_to_s3(
            session, MODEL_PKL_FILENAME, inputs[0].BUCKET_NAME, folder=MODELS_FOLDER
        )

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
