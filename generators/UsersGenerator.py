# pylint: disable=C0415,C0103
class UsersGenerator:
    def __init__(self, out_users_filename, in_users_filenames, user_count) -> None:
        self.out_users_filename = out_users_filename
        self.in_users_filenames = in_users_filenames
        self.user_count = user_count

    def generate(self):
        """
        Users Data Generation
        """
        import generators.datagenerator.users as users
        from generators.datagenerator.users import UserPool
        import numpy as np
        import random
        import json
        import pandas as pd
        import gzip
        from pathlib import Path

        Path(self.out_users_filename).parents[0].mkdir(parents=True, exist_ok=True)
        users.Faker.seed(42)  # Deterministic randomness
        random.seed(42)  # Deterministic randomness
        np.random.seed(42)  # Deterministic randomness

        num_users = self.user_count
        num_cstore_users = int(num_users / 10)
        num_web_users = num_users - num_cstore_users

        print(f"Generating {num_web_users} random web users...")

        pool = UserPool.new_file(
            self.in_users_filenames[0],
            num_web_users,
            category_preference_personas=users.category_preference_personas_web,
        )
        pool_check = UserPool.from_file(self.in_users_filenames[0])

        if repr(pool.users) != repr(pool_check.users):
            raise ValueError("User generation: re-loading users alters something.")

        print(f"Generating {num_cstore_users} random c-store users...")

        cstore_pool = UserPool.new_file(
            self.in_users_filenames[1],
            num_cstore_users,
            category_preference_personas=users.category_preference_personas_cstore,
            selectable_user=False,
            start_user=num_web_users,
        )
        cstore_pool_check = UserPool.from_file(self.in_users_filenames[1])

        if repr(cstore_pool.users) != repr(cstore_pool_check.users):
            raise ValueError("User generation: re-loading users alters something.")

        # User info is stored in the repository - it was automatically generated
        users = []
        for in_users_filename in self.in_users_filenames:
            with gzip.open(in_users_filename, "r") as f:
                users += json.load(f)

        users_df = pd.DataFrame(users)

        users_dataset_df = users_df[["id", "age", "gender"]]
        users_dataset_df = users_dataset_df.rename(
            columns={"id": "USER_ID", "age": "AGE", "gender": "GENDER"}
        )

        users_dataset_df.to_csv(self.out_users_filename, index=False)
        print(" UsersGenerator Done")
        return users_df
