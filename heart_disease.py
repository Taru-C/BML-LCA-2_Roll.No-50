import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


data = pd.read_csv(url, header=None)
data.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

data = data.replace("?", pd.NA)
data = data.dropna()
data = data.astype(float)


X = data.drop("target", axis=1)
y = (data["target"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Classifier:")
print("Dataset shape:", data.shape)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")