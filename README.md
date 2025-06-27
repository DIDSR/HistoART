
# HistoART: Artifact Detection and Analysis in Digital Pathology

<p align="center">
  <h1 align="center">HistoART</h1>
</p>

<p align="center">
  <h3 align="center">Artifact Detection Toolbox</h3>
</p>

---

## General Information
**`HistoArt`** is a Python-based toolkit designed for researchers and professionals working with digital pathology images. Its primary goal is to facilitate effective detection, classification, and quantification of artifacts in a histopathological slide dataset through both deep learning and hand-crafted feature approaches. This toolkit is particularly aimed at pathologists, data scientists, and students looking to streamline and enhance their analysis workflows.

We encourage open-source contributions and collaborations. For further information or contributions, please contact the repository maintainer.

---

## Modules

HistoArt comprises several integrated modules:

1. **Dataset Handling (`datasets.py`):**
   - Dataset loading and preprocessing for clean and artifact-labeled images.
   - Management of combined dataset classes for training and validation, for both deep learning and hand-crafted feature based models

2. **Feature Extraction (`analysis.py`):**
   - Prints the percentage of artifacts and artifact free images in your dataset

3. **Visualization (`visualize.py`):**
   - Tools for visualizing histograms, boxplots, and other statistical representations of image features and artifacts.
   - Interactive tools for exploring distributions and feature correlations.

4. **Model Execution (`model_execution.py`):**
   - Handles logic for executing the three core models of this tool: Foundation, Deep Learning, and Knowledge-Based

5. **Metrics and Performance Assessment (`metrics.ipynb`):**
   - Evaluation classification metrics such as accuracy, recall, precision, F1, and AUC to assess model performance.

6. **End-to-End Pipeline (`histoart.ipynb`):**
   - Loads and preprocesses images
   - Implementation and execution of classification models for artifact detection.
   - Prints analysis

---

## Installation

To set up the HistoArt environment, first clone this repository and navigate to the project directory:

```bash
git clone https://github.com/mousavikahaki/HistoART.git
cd HistoArt
```

Create a virtual environment and install dependencies from the provided `requirements.txt`:

```bash
python3 -m venv histoart_env
source histoart_env/bin/activate
pip install -r requirements.txt
```

**Tested Environment:**
- Linux (Ubuntu 22.04 LTS recommended)
- Python 3.10+

### Dependencies

Some key dependencies include:

```sh
numpy==2.1.2
opencv-python==4.11.0.86
scikit-image==0.25.2
scikit-learn==1.6.1
matplotlib==3.10.1
pyfeats==1.0.1
mahotas==1.4.18
torch==2.5.1
torchvision==0.20.1
```

(See `requirements.txt` for the full list.)

---

## Getting Started Examples

Several Jupyter notebooks and scripts are provided to quickly familiarize you with the capabilities and usage of HistoArt:

1. [Dataset Preparation and Loading (`utils/datasets.py`)](https://github.com/mousavikahaki/HistoART/blob/main/utils/datasets.py)
2. [Artifact Feature Extraction (`utils/analysis.py`)](https://github.com/mousavikahaki/HistoART/blob/main/utils/analysis.py)
3. [Model Execution and Evaluation (`utils/model_execution.py`)](https://github.com/mousavikahaki/HistoART/blob/main/utils/model_execution.py)
4. [Metrics Calculation (`metrics.ipynb`)](https://github.com/mousavikahaki/HistoART/blob/main/metrics.ipynb)
5. [Visualization Examples (`utils/visualize.py`)](https://github.com/mousavikahaki/HistoART/blob/main/utils/visualize.py)
6. [End-to-End Artifact Analysis (`histoart.ipynb`)](https://github.com/mousavikahaki/HistoART/blob/main/histoart.ipynb)

---

## How to Cite

If you utilize HistoArt in your research or applications, please cite the repository:

```bibtex
@misc{HistoArt2025,
  author = {Seyed M. Kahaki, Alexander R. Webber},
  title = {HistoArt},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/DIDSR/HistoArt}},
}
```

---

## Auxiliary Files and Data

### Suggested open source datasets:
- HistoArt (https://zenodo.org/records/10809442)
- TCGA@Focus (https://zenodo.org/records/3910757)

### Included Models (via Git LFS)
- FMA Binary and Multiclass
- DLA Binary and Multiclass
- KBA Binary (Internal and External) and Multiclass

---

## Contact and Contributions

For any inquiries, suggestions, or collaborative opportunities, please contact Seyed Kahaki or Alex Webber either via this GitHub repo or via email (seyed.kahaki@fda.hhs.gov;Alexander.Webber@fda.hhs.gov).

We warmly welcome pull requests and issues to enhance the project's capabilities and documentation.

---

## Acknowledgments
This project was supported in part by an appointment to the ORISE Research Participation Program at the Center for Devices and Radiological Health, U.S. Food and Drug Administration, administered by the Oak Ridge Institute for Science and Education through an interagency agreement between the U.S. Department of Energy and FDA/CDRH.
