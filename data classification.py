# ============================================================
#             DATA CLASSIFICATION USING AI
# ============================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


print("=" * 55)
print("          DATA CLASSIFICATION USING AI")
print("=" * 55)


# 1. Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("\n1. DATASET LOADED SUCCESSFULLY")
print("Total records:", len(X))
print("Features:", iris.feature_names)


# 2. Display Dataset
data = pd.DataFrame(
    X,
    columns=iris.feature_names
)

data["target"] = y

print("\n2. DATASET PREVIEW")
print(data.head())


# 3. Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n3. DATA SPLIT")
print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# 4. Create Classification Model
model = KNeighborsClassifier(n_neighbors=3)


# 5. Train Model
model.fit(X_train, y_train)

print("\n4. MODEL TRAINED SUCCESSFULLY")


# 6. Make Predictions
y_pred = model.predict(X_test)


# 7. Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n5. MODEL EVALUATION")
print("Accuracy:", round(accuracy * 100, 2), "%")


# 8. Predict New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\n6. NEW DATA PREDICTION")
print("Input:", new_flower)
print("Predicted Flower:",
      iris.target_names[prediction[0]])


print("\n" + "=" * 55)
print("          CLASSIFICATION COMPLETED")
print("=" * 55)