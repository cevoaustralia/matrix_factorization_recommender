# pylint: disable=C0415,C0103
class ItemsGenerator:
    def __init__(
        self,
        out_items_filename,
        in_products_filename,
    ) -> None:
        self.out_items_filename = out_items_filename
        self.in_products_filename = in_products_filename

    def generate(self):
        import pandas as pd
        from pathlib import Path
        import yaml

        GENDER_ANY = "Any"
        NOT_PROMOTED = "N"

        Path(self.out_items_filename).parents[0].mkdir(parents=True, exist_ok=True)

        # Product info is stored in the repository
        with open(self.in_products_filename, "r") as f:
            products = yaml.safe_load(f)

        products_df = pd.DataFrame(products)

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
        products_dataset_df.to_csv(self.out_items_filename, index=False)

        print(" ItemsGenerator Done")
        return products_df
