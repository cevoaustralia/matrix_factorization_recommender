import os
import json
import boto3
import pandas as pd
import io
import pickle
from scipy import sparse

NUM_RECOMMENDATIONS = 20


def get_s3_to_df(filename):
    session = boto3.Session()
    obj = session.resource("s3").Object("cevo-mf-recommender", f"fake_data/{filename}")
    body = obj.get()["Body"].read()
    df = pd.read_csv(io.BytesIO(body), encoding="utf8")
    return df


def get_model_from_s3(filename):
    session = boto3.Session()
    obj = session.resource("s3").Object("cevo-mf-recommender", f"models/{filename}")
    body = obj.get()["Body"].read()
    return io.BytesIO(body)


pickled_model = os.environ["BEST_MODEL"]
model = pickle.load(get_model_from_s3(pickled_model))
grouped_df = get_s3_to_df("interactions-confidence.csv")
items_df = get_s3_to_df("items.csv")
data = []


def get_popular_items(grouped_df, N=20):
    """
    Get popular items (to use as baseline)
    """
    popular_items = grouped_df.ITEM_ID.value_counts(sort=True).keys()[:N]
    top_N_popular_items = []
    for item in popular_items:
        item_index = grouped_df.ITEM_IDX.loc[grouped_df.ITEM_ID == item].iloc[0]
        top_N_popular_items.append(item_index)
    return top_N_popular_items


def get_item_details(item_indices):
    items = []
    for item_idx in item_indices:
        # Lookup Item id using item index
        item_id = (
            grouped_df.ITEM_ID.loc[grouped_df.ITEM_IDX == item_idx].iloc[0]
            if len(grouped_df.ITEM_ID.loc[grouped_df.ITEM_IDX == item_idx]) > 0
            else "No item id"
        )
        # Lookup Item description using item id
        item_desc = (
            items_df.PRODUCT_DESCRIPTION.loc[items_df.ITEM_ID == item_id].iloc[0]
            if len(items_df.PRODUCT_DESCRIPTION.loc[items_df.ITEM_ID == item_id]) > 0
            else "No item description"
        )
        # Lookup Item price using item id
        item_price = (
            items_df.PRICE.loc[items_df.ITEM_ID == item_id].iloc[0]
            if len(items_df.PRICE.loc[items_df.ITEM_ID == item_id]) > 0
            else "No item price"
        )
        # Lookup Item category using item id
        item_category = (
            items_df.CATEGORY_L1.loc[items_df.ITEM_ID == item_id].iloc[0]
            if len(items_df.CATEGORY_L1.loc[items_df.ITEM_ID == item_id]) > 0
            else "No item category"
        )
        items.append(
            {
                "description": item_desc,
                "id": item_id,
                "price": item_price,
                "category": item_category,
            }
        )
        # iterate data and update image property
        basedir = "https://cevo-shopping-demo.s3.ap-southeast-2.amazonaws.com/dataset/images/images"
        for product in items:
            product["installments"] = 4
            product["image"] = (
                # if space in category, replace with dash, and omit period in id due to error in prior data generation
                f"{basedir}/{product['category'].replace(' ', '-')}/{product['id']}jpg"
                if " " in product["category"]
                else f"{basedir}/{product['category']}/{product['id']}.jpg"
            )
    return items


def call_recommend(ml_model, user_items, user_idx, num_recomm=NUM_RECOMMENDATIONS):
    """
    Call the recommend function to get the top N recommendations for a user
    """
    recommendations_raw = ml_model.recommend(
        user_idx, user_items[user_idx], N=num_recomm
    )
    predictions = recommendations_raw[0][:num_recomm]

    return predictions


def lambda_handler(event, context):
    print("received event:")
    print(event)

    if event["queryStringParameters"] is not None:
        if event["queryStringParameters"]["user_id"]:
            print("Calling matrix factorization recommender...")
            sparse_user_item = sparse.csr_matrix(
                (
                    grouped_df["CONFIDENCE"].astype(float),
                    (grouped_df["USER_IDX"], grouped_df["ITEM_IDX"]),
                )
            )

            user_index = (
                int(event["queryStringParameters"]["user_id"]) - 1
            )  # user index is just user id less 1
            item_indices = call_recommend(
                model,
                sparse_user_item,
                user_index,
                NUM_RECOMMENDATIONS,
            )
            print(f"item_indices: {item_indices}")
            filteredData = get_item_details(item_indices)
            top_n_data = filteredData[:NUM_RECOMMENDATIONS]
    else:
        print("Just returning popular items...")
        item_indices = get_popular_items(grouped_df)
        items = get_item_details(item_indices)
        top_n_data = items[:NUM_RECOMMENDATIONS]
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": json.dumps(top_n_data, indent=4, sort_keys=True, default=str),
    }
