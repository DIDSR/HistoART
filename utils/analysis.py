import pandas as pd
import matplotlib.pyplot as plt

def print_analysis(csv_path):
    """
    Reads a CSV file containing prediction probabilities and generates a pie chart 
    showing the distribution of "Artifact Free" vs. "Artifact" classifications.

    Args:
        csv_path (str): Path to the CSV file containing a column 'Prediction_Probability'. 
                        Values should be in the range [0, 1], representing the probability 
                        of an image containing an artifact.

    The function:
    - Reads the CSV file into a Pandas DataFrame.
    - Computes the percentage of images classified as "Artifact Free" (probability < 0.5)
      and "Artifact" (probability >= 0.5).
    - Generates a pie chart visualizing the classification distribution.
    """
    
    df     = pd.read_csv(csv_path)
    y_pred = df['Prediction_Probability'].values

    predicted_labels         = (y_pred >= 0.5)
    total_predictions        = len(y_pred)
    artifact_percentage      = (predicted_labels.sum() / total_predictions) * 100
    artifact_free_percentage = 100 - artifact_percentage

    colors  = ['#4CAF50', '#E74C3C']

    explode = (0.1, 0)  # Slightly offset the "Artifact Free" slice
    labels  = ['Artifact Free', 'Artifact']
    sizes   = [artifact_free_percentage, artifact_percentage]

    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=140, explode=explode,
                                       wedgeprops={'edgecolor': 'gray', 'linewidth': 1},
                                       shadow=True)
    
    for text in texts + autotexts:
        text.set_fontsize(12)
    
    plt.title("Artifact Free vs Artifact Distribution")
    plt.show()
