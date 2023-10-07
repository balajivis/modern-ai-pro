#This program would help train a model for predicting heart disease and save the model

import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
def sanity_check(df):
    # Select an 'unhealthy' sample (target = 1)
    sample_unhealthy = df[df['target'] == 1].sample(1).iloc[0]
    print("Unhealthy Sample:")
    print(sample_unhealthy)

    print("\n=================\n")

    # Select a 'healthy' sample (target = 0)
    sample_healthy = df[df['target'] == 0].sample(1).iloc[0]
    print("Healthy Sample:")
    print(sample_healthy)

# Clean the data
def clean_data(df):
    selected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[selected_columns]
    y = df['target']  # Assuming 'target' is the column name for labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test

# Build the model
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(13,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

# Evaluate the model
def eval_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss:.2f}')
    print(f'Accuracy: {accuracy:.2f}')

# Save the model
def save_model(model):
    model.save('heart-disease-model.h5')

df = pd.read_csv("heart.csv")
print(df.head())
sanity_check(df)
X_train, X_test, y_train, y_test = clean_data(df)
model = build_model()
model = train_model(model, X_train, y_train)
eval_model(model, X_test, y_test)
save_model(model)


