import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("best_house_price_model.pkl")

def predict_custom_house(model, feature_dict):
    input_df = pd.DataFrame([feature_dict])
    predicted_price = model.predict(input_df)[0]
    return predicted_price


# Example input (replace values)


new_house = {
    "area": 120,
    "number_rooms": 3,
    "city": "Brussels",
    "floor": 2,
    "parking": "yes",
    "balcony": "no",
    # include ALL remaining columns your model expects
}

price = predict_custom_house(model, new_house)
print(f"Estimated Price: {price:,.2f}")

