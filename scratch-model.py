import pandas as pd
import pickle
from scipy import sparse


def call_recommend(ml_model, user_items, user_idx, N=20):
    """
    Call the recommend function to get the top N recommendations for a user
    """
    num_recomm = N
    recommendations_raw = ml_model.recommend(
        user_idx, user_items[user_idx], N=num_recomm
    )
    predictions = recommendations_raw[0][
        :num_recomm
    ]  # these are the top n preditictions using the train set
    # index = 0
    # for item in recommendations_raw[0][:num_recomm]:
    #     print(f"item idx: {item}, score: {recommendations_raw[1][index]}")
    #     index += 1

    return predictions


grouped_df = pd.read_csv(f"./fake_data/interactions-confidence.csv")
items_df = pd.read_csv(f"./fake_data/items.csv")
# print(items_df.head())

sparse_person_content = sparse.csr_matrix(
    (
        grouped_df["CONFIDENCE"].astype(float),
        (grouped_df["USER_IDX"], grouped_df["ITEM_IDX"]),
    )
)

model = pickle.load(open("mf-recommender-2023-05-19-06-01-53.pkl", "rb"))

item_indices = call_recommend(model, sparse_person_content, 34, N=20)

# print(grouped_df.value_counts("ITEM_IDX"))

for item_idx in item_indices:
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
    print(f"{item_idx} {item_id}, {item_desc}")

    # try with these user indices below
    # user index => persona
    # 24 => footwear, jewellery, furniture
    # 7 => electronics, outdoors, footwear
    # 2 => floral, beauty, jewellery
    # 19 => beauty, accesssories, instruments
    # 34 => housewwares, tools, beauty
