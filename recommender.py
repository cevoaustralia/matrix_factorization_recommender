import io
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

        self.next(self.data_generation_users)

    @step
    def data_generation_users(self):
        """
        Generates list of random users using the datagenerator library included in
        this directory.

        The random users are written to a gzipped JSON file which, when copied
        into place, will be loaded by the Users microservice during initialization.

        This script only needs to be run once to produce a random users data
        file that is bundled with all Retail Demo Store deployments.
        """
        import generators.datagenerator.users as users
        from generators.datagenerator.users import UserPool

        import numpy as np
        import random

        users.Faker.seed(42)  # Deterministic randomness
        random.seed(42)  # Deterministic randomness
        np.random.seed(42)  # Deterministic randomness

        num_users = 10000
        num_cstore_users = int(num_users / 10)
        num_web_users = num_users - num_cstore_users

        print("Generating {} random web users...".format(num_web_users))

        pool = UserPool.new_file(
            "./fake_data/users.json.gz",
            num_web_users,
            category_preference_personas=users.category_preference_personas_web,
        )
        pool_check = UserPool.from_file("./fake_data/users.json.gz")

        if pool.users.__repr__() != pool_check.users.__repr__():
            raise ValueError("User generation: re-loading users alters something.")

        print("Generating {} random c-store users...".format(num_cstore_users))

        cstore_pool = UserPool.new_file(
            "./fake_data/cstore_users.json.gz",
            num_cstore_users,
            category_preference_personas=users.category_preference_personas_cstore,
            selectable_user=False,
            start_user=num_web_users,
        )
        cstore_pool_check = UserPool.from_file("./fake_data/cstore_users.json.gz")

        if cstore_pool.users.__repr__() != cstore_pool_check.users.__repr__():
            raise ValueError("User generation: re-loading users alters something.")

        print("Done")

        self.next(self.data_generation_items)

    def generate_user_items(
        self,
        out_users_filename,
        out_items_filename,
        in_users_filenames,
        in_products_filename,
    ):
        import json
        import pandas as pd
        from pathlib import Path
        import gzip
        import yaml

        GENDER_ANY = "Any"
        NOT_PROMOTED = "N"

        Path(out_items_filename).parents[0].mkdir(parents=True, exist_ok=True)
        Path(out_users_filename).parents[0].mkdir(parents=True, exist_ok=True)

        # Product info is stored in the repository
        with open(in_products_filename, "r") as f:
            products = yaml.safe_load(f)

        products_df = pd.DataFrame(products)

        # User info is stored in the repository - it was automatically generated
        users = []
        for in_users_filename in in_users_filenames:
            with gzip.open(in_users_filename, "r") as f:
                users += json.load(f)

        users_df = pd.DataFrame(users)

        products_dataset_df = products_df[
            [
                "id",
                "price",
                "category",
                "style",
                "description",
                "gender_affinity",
                "promoted",
            ]
        ]
        products_dataset_df = products_dataset_df.rename(
            columns={
                "id": "ITEM_ID",
                "price": "PRICE",
                "category": "CATEGORY_L1",
                "style": "CATEGORY_L2",
                "description": "PRODUCT_DESCRIPTION",
                "gender_affinity": "GENDER",
                "promoted": "PROMOTED",
            }
        )
        # Since GENDER column requires a value for all rows, default all nulls to "Any"
        products_dataset_df["GENDER"].fillna(GENDER_ANY, inplace=True)
        products_dataset_df.loc[
            products_dataset_df["PROMOTED"] == True, "PROMOTED"
        ] = "Y"
        products_dataset_df["PROMOTED"].fillna(NOT_PROMOTED, inplace=True)
        products_dataset_df.to_csv(out_items_filename, index=False)

        users_dataset_df = users_df[["id", "age", "gender"]]
        users_dataset_df = users_dataset_df.rename(
            columns={"id": "USER_ID", "age": "AGE", "gender": "GENDER"}
        )

        users_dataset_df.to_csv(out_users_filename, index=False)

        return users_df, products_df

    @step
    def data_generation_items(self):
        """
        This script exists so that when developing or internal deployment of public commits
        the new Personalize training files can be generated, picked up, and uploaded.

        This script generates interactions for Amazon Personalize by heuristic simulation. It is based off the notebook
        under workshop/01-Personalization where the logic is explained in more detail.
        However, it has been improved in the following ways:
        1. This script is deterministic; random seeds from RANDOM_SEED random variable below.
        2. Logic exists for ensuring balance across categories.
        3. Logic exists for ensuring balance across products.
        4. Discount events are also generated according to 3 different types of users: discount-likers discount-indifferent,
            and price-sensitive-discount-likers.
        Item 1 allows us to re-generate data during staging and item 2 and 3 helps recommendations look appropriate in
        the final demo. If there is poor balance across products and categories then one may not get recommendations
        for products in the same category. This is a hotfix for the logic whereby we generate profiles and probabilistically
        sample product categories according to the sample user profile. Item 4 is necessary for training the discounts
        personalizeation campaign.
        """
        IN_PRODUCTS_FILENAME = "./generators/products.yaml"
        IN_USERS_FILENAMES = [
            "./fake_data/users.json.gz",
            "./fake_data/cstore_users.json.gz",
        ]

        # This is where stage.sh will pick them up from
        out_items_filename = f"./fake_data/items.csv"
        out_users_filename = f"./fake_data/users.csv"

        self.users_df, self.products_df = self.generate_user_items(
            out_users_filename,
            out_items_filename,
            IN_USERS_FILENAMES,
            IN_PRODUCTS_FILENAME,
        )
        self.next(self.data_generation_interactions)

    def generate_interactions(self, out_interactions_filename, users_df, products_df):
        """Generate items.csv, users.csv from users and product dataframes makes interactions.csv by simulating some
        shopping behaviour."""
        import numpy as np
        import time
        import csv
        from pathlib import Path
        import random
        from collections import defaultdict
        from datetime import datetime, timedelta

        # Keep things deterministic
        RANDOM_SEED = 0

        # Where to put the generated data so that it is picked up by stage.sh
        # GENERATED_DATA_ROOT = "src/aws-lambda/personalize-pre-create-resources/data"

        # Interactions will be generated across the last 90 days
        DAYS_BACK = 90
        now = datetime.now()
        LAST_TIMESTAMP = int(datetime.timestamp(now))
        FIRST_TIMESTAMP = int(datetime.timestamp(now - timedelta(days=DAYS_BACK)))

        # Users are set up with 3 product categories on their personas. If [0.6, 0.25, 0.15] it means
        # 60% of the time they'll choose a product from the first category, etc.
        CATEGORY_AFFINITY_PROBS = [0.6, 0.25, 0.15]

        # After a product, there are this many products within the category that a user is likely to jump on next.
        # The purpose of this is to keep recommendations focused within the category if there are too many products
        # in a category, because at present the user profiles approach samples products from a category.
        PRODUCT_AFFINITY_N = 4

        # from 0 to 1. If 0 then products in busy categories get represented less. If 1 then all products same amount.
        NORMALISE_PER_PRODUCT_WEIGHT = 1.0

        # With this probability a product interaction will be with the product discounted
        # Here we go the other way - what is the probability that a product that a user is already interacting
        # with is discounted - depending on whether user likes discounts or not
        DISCOUNT_PROBABILITY = 0.2
        DISCOUNT_PROBABILITY_WITH_PREFERENCE = 0.5

        PROGRESS_MONITOR_SECONDS_UPDATE = 30

        # The meaning of the below constants is described in the relevant notebook.

        # Minimum number of interactions to generate
        min_interactions = 700000
        # min_interactions = 50000

        # Percentages of each event type to generate
        product_added_percent = 0.08
        cart_viewed_percent = 0.05
        checkout_started_percent = 0.02
        order_completed_percent = 0.01

        # Count of interactions generated for each event type
        product_viewed_count = 0
        discounted_product_viewed_count = 0
        product_added_count = 0
        discounted_product_added_count = 0
        cart_viewed_count = 0
        discounted_cart_viewed_count = 0
        checkout_started_count = 0
        discounted_checkout_started_count = 0
        order_completed_count = 0
        discounted_order_completed_count = 0

        Path(out_interactions_filename).parents[0].mkdir(parents=True, exist_ok=True)

        # ensure determinism
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        start_time_progress = int(time.time())
        next_timestamp = FIRST_TIMESTAMP
        seconds_increment = int((LAST_TIMESTAMP - FIRST_TIMESTAMP) / min_interactions)
        next_update_progress = start_time_progress + PROGRESS_MONITOR_SECONDS_UPDATE / 2

        average_product_price = int(products_df.price.mean())
        print("Average product price: ${:.2f}".format(average_product_price))

        if seconds_increment <= 0:
            raise AssertionError(f"Should never happen: {seconds_increment} <= 0")

        print("Minimum interactions to generate: {}".format(min_interactions))
        print(
            "Starting timestamp: {} ({})".format(
                next_timestamp,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(next_timestamp)),
            )
        )
        print("Seconds increment: {}".format(seconds_increment))

        print("Generating interactions... (this may take a few minutes)")
        interactions = 0

        subsets_cache = {}

        user_to_product = defaultdict(set)

        category_affinity_probs = np.array(CATEGORY_AFFINITY_PROBS)

        print("Writing interactions to: {}".format(out_interactions_filename))

        with open(out_interactions_filename, "w") as outfile:
            f = csv.writer(outfile)
            f.writerow(["ITEM_ID", "USER_ID", "EVENT_TYPE", "TIMESTAMP", "DISCOUNT"])

            category_frequencies = products_df.category.value_counts()
            category_frequencies /= sum(category_frequencies.values)

            interaction_product_counts = defaultdict(int)

            # Here we build up a list for each category/gender, of product
            # affinities. The product affinity is keyed by one product,
            # so we do not end up with exactly PRODUCT_AFFINITY_N sized
            # cliques. They overlap a little over multiple users
            # - that is why PRODUCT_AFFINITY_N
            # can be a little bit lower than a desired clique size.
            all_categories = products_df.category.unique()
            product_affinities_bycatgender = {}
            for category in all_categories:
                for gender in ["M", "F"]:
                    products_cat = products_df.loc[products_df.category == category]
                    products_cat = products_cat.loc[
                        products_cat.gender_affinity.isnull()
                        | (products_cat.gender_affinity == gender)
                    ].id.values
                    # We ensure that all products have PRODUCT_AFFINITY_N products that lead into it
                    # and PRODUCT_AFFINITY_N products it leads to
                    affinity_matrix = sum(
                        [
                            np.roll(np.identity(len(products_cat)), [0, i], [0, 1])
                            for i in range(PRODUCT_AFFINITY_N)
                        ]
                    )
                    np.random.shuffle(affinity_matrix)
                    affinity_matrix = affinity_matrix.T
                    np.random.shuffle(affinity_matrix)
                    affinity_matrix = affinity_matrix.astype(
                        bool
                    )  # use as boolean index
                    affinity_matrix = affinity_matrix | np.identity(
                        len(products_cat), dtype=bool
                    )

                    product_infinities = [products_cat[row] for row in affinity_matrix]
                    product_affinities_bycatgender[(category, gender)] = {
                        products_cat[i]: products_df.loc[
                            products_df.id.isin(product_infinities[i])
                        ]
                        for i in range(len(products_cat))
                    }

            user_category_to_first_prod = {}

            while interactions < min_interactions:
                if time.time() > next_update_progress:
                    rate = interactions / (time.time() - start_time_progress)
                    to_go = (min_interactions - interactions) / rate
                    print(
                        "Generated {} interactions so far (about {} seconds to go)".format(
                            interactions, int(to_go)
                        )
                    )
                    next_update_progress += PROGRESS_MONITOR_SECONDS_UPDATE

                # Pick a random user
                user = users_df.loc[random.randint(0, users_df.shape[0] - 1)]

                # Determine category affinity from user's persona
                persona = user["persona"]
                # If user persona has sub-categories, we will use those sub-categories to find products for users to partake
                # in interactions with. Otehrwise, we will use the high-level categories.
                has_subcategories = ":" in user["persona"]
                preferred_categories_and_subcats = persona.split("_")
                preferred_highlevel_categories = [
                    catstring.split(":")[0]
                    for catstring in preferred_categories_and_subcats
                ]
                # preferred_styles = [catstring.split(':')[1] for catstring in preferred_categories_and_subcats]

                p_normalised = (
                    category_affinity_probs
                    * category_frequencies[preferred_highlevel_categories].values
                )
                p_normalised /= p_normalised.sum()
                p = (
                    NORMALISE_PER_PRODUCT_WEIGHT * p_normalised
                    + (1 - NORMALISE_PER_PRODUCT_WEIGHT) * category_affinity_probs
                )

                # Select category based on weighted preference of category order.
                chosen_category_ind = np.random.choice(
                    list(range(len(preferred_categories_and_subcats))), 1, p=p
                )[0]
                category = preferred_highlevel_categories[chosen_category_ind]
                # category_and_subcat = np.random.choice(preferred_categories_and_subcats, 1, p=p)[0]

                discount_persona = user["discount_persona"]

                gender = user["gender"]

                if has_subcategories:
                    # if there is a preferred style we choose from those products with this style and category
                    # but we ignore gender.
                    # We also do not attempt to keep balance across categories.
                    style = preferred_categories_and_subcats[chosen_category_ind].split(
                        ":"
                    )[1]
                    cachekey = ("category-style", category, style)
                    prods_subset_df = subsets_cache.get(cachekey)

                    if prods_subset_df is None:
                        # Select products from selected category without gender affinity or that match user's gender
                        prods_subset_df = products_df.loc[
                            (products_df["category"] == category)
                            & (products_df["style"] == style)
                        ]
                        # Update cache for quicker lookup next time
                        subsets_cache[cachekey] = prods_subset_df
                else:
                    # We are only going to use the machinery to keep things balanced
                    # if there is no style appointed on the user preferences.
                    # Here, in order to keep the number of products that are related to a product,
                    # we restrict the size of the set of products that are recommended to an individual
                    # user - in effect, the available subset for a particular category/gender
                    # depends on the first product selected, which is selected as per previous logic
                    # (looking at category affinities and gender)
                    usercat_key = (
                        user["id"],
                        category,
                    )  # has this user already selected a "first" product?
                    if usercat_key in user_category_to_first_prod:
                        # If a first product is already selected, we use the product affinities for that product
                        # To provide the list of products to select from
                        first_prod = user_category_to_first_prod[usercat_key]
                        prods_subset_df = product_affinities_bycatgender[
                            (category, gender)
                        ][first_prod]

                    if not usercat_key in user_category_to_first_prod:
                        # If the user has not yet selected a first product for this category
                        # we do it by choosing between all products for gender.

                        # First, check if subset data frame is already cached for category & gender
                        cachekey = ("category-gender", category, gender)
                        prods_subset_df = subsets_cache.get(cachekey)
                        if prods_subset_df is None:
                            # Select products from selected category without gender affinity or that match user's gender
                            prods_subset_df = products_df.loc[
                                (products_df["category"] == category)
                                & (
                                    (products_df["gender_affinity"] == gender)
                                    | (products_df["gender_affinity"].isnull())
                                )
                            ]
                            # Update cache
                            subsets_cache[cachekey] = prods_subset_df

                # Pick a random product from gender filtered subset
                product = prods_subset_df.sample().iloc[0]

                interaction_product_counts[product.id] += 1

                user_to_product[user["id"]].add(product["id"])

                if not usercat_key in user_category_to_first_prod:
                    user_category_to_first_prod[usercat_key] = product["id"]

                # Decide if the product the user is interacting with is discounted
                if discount_persona == "discount_indifferent":
                    discounted = random.random() < DISCOUNT_PROBABILITY
                elif discount_persona == "all_discounts":
                    discounted = random.random() < DISCOUNT_PROBABILITY_WITH_PREFERENCE
                elif discount_persona == "lower_priced_products":
                    if product.price < average_product_price:
                        discounted = (
                            random.random() < DISCOUNT_PROBABILITY_WITH_PREFERENCE
                        )
                    else:
                        discounted = random.random() < DISCOUNT_PROBABILITY
                else:
                    raise ValueError(
                        f"Unable to handle discount persona: {discount_persona}"
                    )

                num_interaction_sets_to_insert = 1
                prodcnts = list(interaction_product_counts.values())
                prodcnts_max = max(prodcnts) if len(prodcnts) > 0 else 0
                prodcnts_min = min(prodcnts) if len(prodcnts) > 0 else 0
                prodcnts_avg = sum(prodcnts) / len(prodcnts) if len(prodcnts) > 0 else 0
                if interaction_product_counts[product.id] * 2 < prodcnts_max:
                    num_interaction_sets_to_insert += 1
                if interaction_product_counts[product.id] < prodcnts_avg:
                    num_interaction_sets_to_insert += 1
                if interaction_product_counts[product.id] == prodcnts_min:
                    num_interaction_sets_to_insert += 1

                for _ in range(num_interaction_sets_to_insert):
                    discount_context = "Yes" if discounted else "No"

                    this_timestamp = next_timestamp + random.randint(
                        1, seconds_increment
                    )
                    f.writerow(
                        [
                            product["id"],
                            user["id"],
                            "View",
                            this_timestamp,
                            discount_context,
                        ]
                    )

                    next_timestamp += seconds_increment
                    product_viewed_count += 1
                    interactions += 1

                    if discounted:
                        discounted_product_viewed_count += 1

                    if product_added_count < int(
                        product_viewed_count * product_added_percent
                    ):
                        this_timestamp += random.randint(1, int(seconds_increment / 2))
                        f.writerow(
                            [
                                product["id"],
                                user["id"],
                                "AddToCart",
                                this_timestamp,
                                discount_context,
                            ]
                        )
                        interactions += 1
                        product_added_count += 1

                        if discounted:
                            discounted_product_added_count += 1

                    if cart_viewed_count < int(
                        product_viewed_count * cart_viewed_percent
                    ):
                        this_timestamp += random.randint(1, int(seconds_increment / 2))
                        f.writerow(
                            [
                                product["id"],
                                user["id"],
                                "ViewCart",
                                this_timestamp,
                                discount_context,
                            ]
                        )
                        interactions += 1
                        cart_viewed_count += 1
                        if discounted:
                            discounted_cart_viewed_count += 1

                    if checkout_started_count < int(
                        product_viewed_count * checkout_started_percent
                    ):
                        this_timestamp += random.randint(1, int(seconds_increment / 2))
                        f.writerow(
                            [
                                product["id"],
                                user["id"],
                                "StartCheckout",
                                this_timestamp,
                                discount_context,
                            ]
                        )
                        interactions += 1
                        checkout_started_count += 1
                        if discounted:
                            discounted_checkout_started_count += 1

                    if order_completed_count < int(
                        product_viewed_count * order_completed_percent
                    ):
                        this_timestamp += random.randint(1, int(seconds_increment / 2))
                        f.writerow(
                            [
                                product["id"],
                                user["id"],
                                "Purchase",
                                this_timestamp,
                                discount_context,
                            ]
                        )
                        interactions += 1
                        order_completed_count += 1
                        if discounted:
                            discounted_order_completed_count += 1

        print("Interactions generation done.")
        print(f"Total interactions: {interactions}")
        print(
            f"Total product viewed: {product_viewed_count} ({discounted_product_viewed_count})"
        )
        print(
            f"Total product added: {product_added_count} ({discounted_product_added_count})"
        )
        print(
            f"Total cart viewed: {cart_viewed_count} ({discounted_cart_viewed_count})"
        )
        print(
            f"Total checkout started: {checkout_started_count} ({discounted_checkout_started_count})"
        )
        print(
            f"Total order completed: {order_completed_count} ({discounted_order_completed_count})"
        )

        globals().update(
            locals()
        )  # This can be used for inspecting in console after script ran or if run with ipython.
        print("Generation script finished")

    @step
    def data_generation_interactions(self):
        # This is where stage.sh will pick them up from
        self.out_interactions_filename = f"./fake_data/interactions.csv"
        self.generate_interactions(
            self.out_interactions_filename, self.users_df, self.products_df
        )
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
