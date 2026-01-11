
# Imports
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



# Load dataset

diabetes = pd.read_csv("diabetes.csv")

# Optional sanity check
assert diabetes.isnull().sum().sum() == 0, "Dataset contains missing values"


# Features & labels
X = diabetes.iloc[:, :-1].values
y = diabetes.iloc[:, -1].values



# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234,
    stratify=y
)



# Build model
model = Sequential([
    Dense(
        24,
        input_shape=(X_train.shape[1],),
        activation="relu",
        kernel_initializer="random_normal"
    ),
    Dense(
        12,
        activation="relu",
        kernel_initializer="random_normal"
    ),
    Dense(1, activation="sigmoid")
])


# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)


# Train model
model.fit(
    X_train,
    y_train,
    epochs=160,
    batch_size=10,
    verbose=0
)


# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")


# Predictions (TF 2.x way)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)


# Metrics
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
