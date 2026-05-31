"""Mine smartphone-to-accessory cross-sell association rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run(input_path: Path) -> None:
    df = pd.read_csv(input_path)
    required = {"Transaction_ID", "Product_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Basket data missing columns: {sorted(missing)}")
    type_col = "Product_Type" if "Product_Type" in df.columns else None
    basket = (
        df.assign(value=1)
        .pivot_table(index="Transaction_ID", columns="Product_ID", values="value", aggfunc="max", fill_value=0)
        .astype(bool)
    )
    itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.3)
    rules = rules[rules["lift"] >= 1.2].copy()
    if type_col:
        product_types = df.drop_duplicates("Product_ID").set_index("Product_ID")[type_col].to_dict()

        def is_smartphone_to_accessory(row) -> bool:
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            return any(product_types.get(item) == "smartphone" for item in antecedents) and any(
                product_types.get(item) == "accessory" for item in consequents
            )

        rules = rules[rules.apply(is_smartphone_to_accessory, axis=1)]
    rules["antecedent"] = rules["antecedents"].map(lambda s: ",".join(sorted(s)))
    rules["consequent"] = rules["consequents"].map(lambda s: ",".join(sorted(s)))
    out = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
    Path("data-project/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data-project/processed/association_rules.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data-project/raw/baskets.csv"))
    args = parser.parse_args()
    run(args.input)
