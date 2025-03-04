import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("balanced_dataset.csv")
sensor_columns = df.columns[:-1]  # All sensor columns
label_column = "label"  # Classification label

# Train a Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(df[sensor_columns])

def classify_input(sensor_data):
    input_values = [list(sensor_data.values())]
    distances, indices = knn.kneighbors(input_values)

    # Find the most common label among nearest neighbors
    nearest_labels = df.iloc[indices[0]][label_column].values
    classification = max(set(nearest_labels), key=list(nearest_labels).count)

    # Get similar cases
    similar_cases = df.iloc[indices[0]].to_dict(orient="records")

    return classification, similar_cases
