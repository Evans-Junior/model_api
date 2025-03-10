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


import numpy as np

def classify_input(sensor_data):
    try:
        # Expected sensor keys
        sensor_order = ["SP-3", "MQ-3", "TGS 822", "MQ-138", "MQ-137", "TGS 813", "TGS-800", "MQ-135"]
        
        # Debugging: Print received sensor keys
        print(f"Received Sensor Keys: {list(sensor_data.keys())}")

        # Validate if all required sensors are present
        missing_sensors = [s for s in sensor_order if s not in sensor_data]
        if missing_sensors:
            return "error", f"Missing sensor data: {missing_sensors}"

        # Convert to NumPy array
        input_values = np.array([[sensor_data[sensor] for sensor in sensor_order]], dtype=float)

        print(f"Processed Input Shape: {input_values.shape}")  # Should be (1, 8)

        # Run KNN classification
        distances, indices = knn.kneighbors(input_values)
        similar_cases = df.iloc[indices[0]]

        classification = similar_cases["label"].mode()[0]  # Most common label

        return classification, similar_cases.to_dict(orient="records")
    
    except Exception as e:
        return "error", f"Error processing input: {str(e)}"
