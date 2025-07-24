import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV with proper encoding
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Clean 'Duration' column
df['Duration'] = df['Duration'].str.replace(' min', '', regex=False).astype(float)

# Clean 'Votes' column (e.g., '$5.16M' → 5160000)
def convert_votes(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('$', '').replace('M', '')
        try:
            return float(value) * 1_000_000
        except:
            return None
    return value

df['Votes'] = df['Votes'].apply(convert_votes)

# Drop rows with missing values
df = df.dropna(subset=['Rating', 'Duration', 'Votes'])

# Prepare features and label
X = df[['Duration', 'Votes']]
y = df['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Output result
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save result
with open("output.txt", "w") as f:
    f.write("Movie Rating Prediction Output\n")
    f.write("------------------------------\n")
    f.write(f"Mean Squared Error: {mse:.2f}\n")
    f.write(f"R2 Score: {r2:.2f}\n")
