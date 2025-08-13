# iris_flower_classification.py

# Step 1: Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 2: Load Dataset (make sure iris.csv is in the same folder)
df = pd.read_csv("iris.csv")

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Separate Features and Target
X = df.iloc[:, :-1]  # all columns except last
y = df.iloc[:, -1]   # last column (species)

# Step 4: Visualize Data
print("\nGenerating Pairplot...")
sns.pairplot(df, hue='species', palette='husl')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
