import pandas as pd
import numpy as np

# 1. Load dataset

df = pd.read_csv("team-cleaned_data.csv")

print("Original shape:", df.shape)



# 2. Drop rows with missing PRICE

df = df.dropna(subset=["price"])


# 3. Remove rows with missing locality_name (only 2 rows)

df = df.dropna(subset=["locality_name"])


# 4. Create NEW engineered columns seen in cleaned_output.csv: province, property_type_name, state_mapped, region, has_garden


# province: comes from postal_code (Belgian mapping example)
def map_province(pc):
    pc = int(pc)
    if 1000 <= pc <= 1299:
        return "brussels"
    elif 1300 <= pc <= 1499:
        return "walloon brabant"
    elif 1500 <= pc <= 1999:
        return "flemish brabant"
    elif 2000 <= pc <= 2999:
        return "antwerp"
    elif 3000 <= pc <= 3499:
        return "flemish brabant"
    elif 3500 <= pc <= 3999:
        return "limburg"
    elif 4000 <= pc <= 4999:
        return "liege"
    elif 5000 <= pc <= 5999:
        return "namur"
    elif 6000 <= pc <= 6599:
        return "hainaut"
    elif 6600 <= pc <= 6999:
        return "luxembourg"
    else:
        return "unknown"

df["province"] = df["postal_code"].apply(map_province)


# property_type_name → simplified class extracted from property_type
df["property_type_name"] = df["property_type"].str.lower().replace({
    "house": "house",
    "villa": "house",
    "apartment": "apartment",
    "penthouse": "apartment",
    "duplex": "apartment",
}).fillna("other")


# state_mapped → cleaning of state column
df["state_mapped"] = df["state"].str.lower().replace({
    "good": "ready_to_move_in",
    "excellent": "ready_to_move_in",
    "to renovate": "to_renovate",
    "to be done up": "to_renovate",
})


# region → derived from province
df["region"] = df["province"].replace({
    "brussels": "brussels",
    "walloon brabant": "wallonia",
    "liege": "wallonia",
    "namur": "wallonia",
    "luxembourg": "wallonia",
    "hainaut": "wallonia",
    "flemish brabant": "flanders",
    "antwerp": "flanders",
    "limburg": "flanders",
})


# has_garden → convert boolean
df["has_garden"] = df["garden"].astype(int)


# 5. Additional filtering: LIMIT ROOMS TO MAX 15


df["number_rooms"] = df["number_rooms"].clip(upper=15)


# 6. Drop rows where categorical enrichments failed. This matches missing counts in cleaned_output.csv


df = df.dropna(subset=["state_mapped", "property_type_name", "province"])


# 7. Reset index


df = df.reset_index(drop=True)

print("Cleaned shape:", df.shape)



# 8. Save final cleaned file


df.to_csv("cleaned_output_rebuilt.csv", index=False)

print("Saved cleaned_output_rebuilt.csv")
