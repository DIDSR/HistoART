import pandas as pd
from utils.uncertainty import Uncertainty_Analysis

def print_analysis(csv_path):
    df = pd.read_csv(csv_path)

    y_pred = df['Prediction_Probability'].values

    # Calculate predicted classes using a threshold of 0.5.
    # Here, a prediction of >= 0.5 is considered "Artifact" and < 0.5 as "Artifact Free".
    predicted_labels = (y_pred >= 0.5)
    total_predictions = len(y_pred)
    artifact_percentage = (predicted_labels.sum() / total_predictions) * 100
    artifact_free_percentage = 100 - artifact_percentage

    print(f"Percentage of predicted Artifact: {artifact_percentage:.2f}%")
    print(f"Percentage of predicted Artifact Free: {artifact_free_percentage:.2f}%")