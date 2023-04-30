class UsersGenerator:
    def __init__(self) -> None:
        pass

    def generate(self):
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

        print(" UsersGenerator Done")
