import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Load data
df = pd.read_csv("data/featured_inventory_data.csv")
X = df.drop(["spoilage_risk_base", "store_id", "item", "date", "expiry_date", "weather", "sustainability_score", "sdg_alignment"], axis=1)
y = df["spoilage_risk_base"]

# Feature selection with increased k to include new features
selector = SelectKBest(score_func=f_regression, k=12)  # Increased to 12 to potentially include new features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Ensure new features are considered if significant
additional_features = ["packaging_waste", "co2_emission_factor", "environmental_impact", "transport_distance_km"]
for feature in additional_features:
    if feature in X.columns and feature not in selected_features:
        scores = selector.scores_[X.columns.get_loc(feature)]
        if scores > selector.scores_.min():  # Add if score is above the lowest selected
            selected_features.append(feature)

# Filter dataframe and save
df = df[selected_features + ["spoilage_risk_base", "store_id", "item", "date", "weather", "sustainability_score", "sdg_alignment"]]
df.rename(columns={"spoilage_risk_base": "spoilage_risk"}, inplace=True)
df.to_csv("data/selected_inventory_data.csv", index=False)
print(f"Selected features: {selected_features}")