import pandas as pd
from utils.uncertainty import Uncertainty_Analysis

def print_analysis(csv_path):
    df = pd.read_csv(csv_path)

    y_pred  = df['Prediction_Probability'].values
    y_truth = df['True_Label'].values

    analysis = Uncertainty_Analysis()
    analysis.perform_Delong = True
    analysis.perform_Bootstrap = True
    analysis.plot_roc = True
    analysis.n_bootstraps = 100
    analysis.tag = 'My Results'

    results  = analysis.get_report(y_pred, y_truth)

    print(results)