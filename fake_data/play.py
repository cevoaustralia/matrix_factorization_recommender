import pandas as pd

df = pd.read_csv("interactions-purchases.csv")

print(df.to_string())

from pandas.api.types import CategoricalDtype
from scipy import sparse
import implicit
from implicit.nearest_neighbours import bm25_weight

# users = df["USER_ID"].unique()
# items = df["ITEM_ID"].unique()
# shape = (len(users), len(items))

# # Create indices for users and items
# user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
# item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
# user_index = df["USER_ID"].astype(user_cat).cat.codes
# item_index = df["ITEM_ID"].astype(item_cat).cat.codes

# # Conversion via COO matrix
# coo = sparse.coo_matrix((df["PURCHASE_COUNT"], (user_index, item_index)), shape=shape)
# csr = coo.tocsr()
# print(csr.toarray())


# # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
# # and to reduce the weight given to popular items
# par_csr = bm25_weight(csr, K1=100, B=0.8)

# # get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
# user_buys = par_csr.T.tocsr()

# # initialize a model
# model = implicit.als.AlternatingLeastSquares(factors=50)

# # train the model on a sparse matrix of user/item/confidence weights
# model.fit(user_buys)

# # recommend items for a user
# recommendations = model.recommend(2, user_buys[2])
# for r in recommendations[0]:
#     print(r)
df["USER_ID"] = df["USER_ID"].astype("category")
df["ITEM_ID"] = df["ITEM_ID"].astype("category")
df["USERID"] = df["USER_ID"].cat.codes
df["ITEMID"] = df["ITEM_ID"].cat.codes

df["PURCHASE_COUNT"] = df["PURCHASE_COUNT"].astype("category")
df["PURCHASE_COUNT"] = df["PURCHASE_COUNT"].cat.codes

sparse_item_user = sparse.csr_matrix(
    (df["PURCHASE_COUNT"].astype("category"), (df["ITEMID"], df["USERID"]))
)
sparse_user_item = sparse.csr_matrix(
    (df["PURCHASE_COUNT"].astype("category"), (df["USERID"], df["ITEMID"]))
)

# Building the model
model = implicit.als.AlternatingLeastSquares(
    factors=20, regularization=0.1, iterations=20
)
alpha_val = 40
data_conf = (sparse_item_user * alpha_val).astype("double")
model.fit(data_conf)

# recommend items for a user
recommendations = model.recommend(1, sparse_item_user[1])
for r in recommendations:
    print(r)
