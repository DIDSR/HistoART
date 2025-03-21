Modules Overview
================

HistoART comprises several integrated modules. Below is an overview of each:

1. **Dataset Handling (`datasets.py`)**

   - Loads and preprocesses digital pathology images.
   - Manages combined dataset classes for training and validation using both deep learning and hand-crafted feature approaches.

2. **Feature Extraction (`analysis.py`)**

   - Computes and prints the percentage of artifacts versus artifact-free images in your dataset.

3. **Visualization (`visualize.py`)**

   - Provides tools for visualizing histograms, boxplots, and other statistical representations.
   - Includes interactive tools for exploring feature distributions and correlations.

4. **Model Execution (`model_execution.py`)**

   - Executes the three core models: Foundation, Deep Learning, and Knowledge-Based.
   
5. **Metrics and Performance Assessment (`metrics.ipynb`)**

   - An interactive notebook for evaluating classification metrics such as accuracy, recall, precision, F1, and AUC.

6. **End-to-End Pipeline (`histoart.ipynb`)**

   - Demonstrates the complete workflow: image loading and preprocessing, model execution, and result analysis.
