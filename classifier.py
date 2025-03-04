import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
# Load dataset
df = pd.read_csv("balanced_dataset.csv")
sensor_columns = df.columns[:-1]  # All sensor columns
label_column = "label"  # Classification label

# Train a Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(df[sensor_columns])

# def classify_input(sensor_data):
#     input_values = [list(sensor_data.values())]
#     distances, indices = knn.kneighbors(input_values)

#     # Find the most common label among nearest neighbors
#     nearest_labels = df.iloc[indices[0]][label_column].values
#     classification = max(set(nearest_labels), key=list(nearest_labels).count)

#     # Get similar cases
#     similar_cases = df.iloc[indices[0]].to_dict(orient="records")

#     return classification, similar_cases


def classify_input(sensor_data):
    input_values = np.array(list(sensor_data.values())).reshape(1, -1)

    # Debugging print: Check if input shape is correct
    print(f"Input shape: {input_values.shape}")  # Should be (1, 8)

    distances, indices = knn.kneighbors(input_values)
    similar_cases = df.iloc[indices[0]]  # Retrieve similar cases
    classification = similar_cases["label"].mode()[0]  # Most common label

    return classification, similar_cases.to_dict()
