import pandas as pd

df = pd.read_csv("interactions.csv")
# count number of rows with same ITEM_ID and USER_ID
df = df.groupby(["ITEM_ID", "USER_ID"]).size().reset_index(name="PURCHASE_COUNT")

# for index, row in df.iterrows():
#     print(row["ITEM_ID"], row["USER_ID"], index)
print(df)
df.to_csv("interactions-purchases.csv", index=False)
