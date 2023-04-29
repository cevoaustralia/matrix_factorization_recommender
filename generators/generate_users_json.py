def generate_users_json():
    # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    # SPDX-License-Identifier: MIT-0

    """
    Generates list of random users using the datagenerator library included in
    this directory.

    The random users are written to a gzipped JSON file which, when copied
    into place, will be loaded by the Users microservice during initialization.

    This script only needs to be run once to produce a random users data
    file that is bundled with all Retail Demo Store deployments.
    """

    import datagenerator.users as users
    from datagenerator.users import UserPool

    import numpy as np
    import random

    users.Faker.seed(42)  # Deterministic randomness
    random.seed(42)  # Deterministic randomness
    np.random.seed(42)  # Deterministic randomness

    num_users = 10000
    num_web_users = (7 * num_users) // 8
    num_cstore_users = num_users - num_web_users
    print(f"num_web_users: {num_web_users}")
    print(f"num_cstore_users: {num_cstore_users}")

    print("Generating {} random web users...".format(num_web_users))

    pool = UserPool.new_file(
        "users.json.gz",
        num_web_users,
        category_preference_personas=users.category_preference_personas_web,
    )
    pool_check = UserPool.from_file("users.json.gz")

    if pool.users.__repr__() != pool_check.users.__repr__():
        raise ValueError("User generation: re-loading users alters something.")

    print("Generating {} random c-store users...".format(num_cstore_users))

    cstore_pool = UserPool.new_file(
        "cstore_users.json.gz",
        num_cstore_users,
        category_preference_personas=users.category_preference_personas_cstore,
        selectable_user=False,
        start_user=num_web_users,
    )
    cstore_pool_check = UserPool.from_file("cstore_users.json.gz")

    if cstore_pool.users.__repr__() != cstore_pool_check.users.__repr__():
        raise ValueError("User generation: re-loading users alters something.")

    print("Done")


if __name__ == "__main__":
    generate_users_json()
