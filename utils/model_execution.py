import torch
import timm
import csv
import pickle
import torch.nn           as nn
import torchvision.models as models
import pandas             as pd
from   tqdm           import tqdm


def save_predictions(epoch, phase, batch_idx, inputs, labels, predicted, probs, output_file):
    """
    Saves model predictions to a CSV file.

    Args:
        epoch (int): Current epoch number.
        phase (str): Training phase (e.g., 'train' or 'test').
        batch_idx (int): Index of the current batch.
        inputs (Tensor): Input images (unused in writing but included for context).
        labels (Tensor): Ground truth labels.
        predicted (Tensor): Predicted labels.
        probs (Tensor): Predicted probabilities.
        output_file (str): Path to the CSV file where predictions will be saved.

    The function appends the predictions to the specified file.
    """

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for i in range(inputs.size(0)):
            row = [
                epoch + 1,
                phase,
                batch_idx * inputs.size(0) + i,
                labels[i].item(),
                predicted[i].item(),
                probs[i].item()
            ]
            writer.writerow(row)

def test_loop(model, loader, epoch, device, output_file, phase):
    """
    Runs a test loop over a DataLoader and saves predictions.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (DataLoader): DataLoader containing the test dataset.
        epoch (int): Current epoch number.
        device (torch.device): Device (CPU or GPU) for computation.
        output_file (str): Path to the CSV file for saving predictions.
        phase (str): Phase name (e.g., 'test').

    The function iterates through the DataLoader, obtains predictions, and saves them.
    """
    loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1} [{phase}]")

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Phase', 'Image_Index', 'True_Label', 'Predicted_Label', 'Prediction_Probability'])
    
    for idx, (inputs, labels) in enumerate(loader_tqdm):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        outputs        = model(inputs)
        predicted      = (torch.sigmoid(outputs) > 0.5).float()
        probs          = torch.sigmoid(outputs)
        
        save_predictions(epoch, phase, idx, inputs, labels, predicted, probs, output_file)

def setup_uni_model():
    """
    Sets up an FMA model from the MahmoodLab repository with a modified classifier head.

    Returns:
        torch.nn.Module: The FMA model with a new linear classification head.

    The function:
    - Loads a pretrained FMA model.
    - Freezes all layers except for the last two blocks.
    - Replaces the classification head with a single-unit linear layer.
    """
    uni = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    
    for name, param in uni.named_parameters():
        if "blocks.11" in name or "blocks.12" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    uni.head = nn.Linear(1024, 1)
    return uni

def setup_resnet_model():
    """
    Sets up a DLA model with a modified classification head.

    Returns:
        torch.nn.Module: The DLA model with a new linear classification head.

    The function:
    - Loads a pretrained DLA model.
    - Freezes all layers except for the last residual block (`layer4`).
    - Replaces the fully connected layer with a single-unit linear layer.
    """
    resnet = models.resnet50(pretrained=True)
    
    for name, param in resnet.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    in_features = resnet.fc.in_features
    resnet.fc   = nn.Linear(in_features, 1)

    return resnet

def runFMA(dataloader, device, model):
    """
    Runs the Fine-tuned Multi-scale Attention (FMA) model on a dataset.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device (CPU or GPU) for computation.
        model (str): Path to the trained model file.

    The function:
    - Loads the pretrained Uni model.
    - Evaluates the model on the provided dataset.
    - Saves predictions to './results/fma_results.csv'.
    """

    fma_model = setup_uni_model()
    fma_model.load_state_dict(torch.load(model, map_location=device))
    fma_model.to(device)
    fma_model.eval()

    test_loop(fma_model, dataloader, 0, device, './results/fma_results.csv', 'test')

def runDLA(dataloader, device, model):
    """
    Runs the Deep Learning Artifact (DLA) model on a dataset.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        device (torch.device): Device (CPU or GPU) for computation.
        model (str): Path to the trained model file.

    The function:
    - Loads the pretrained ResNet-50 model.
    - Evaluates the model on the provided dataset.
    - Saves predictions to './results/dla_results.csv'.
    """
    
    dla_model = setup_resnet_model()
    dla_model.load_state_dict(torch.load(model, map_location=device))
    dla_model.to(device)
    dla_model.eval()

    test_loop(dla_model, dataloader, 0, device, './results/dla_results.csv', 'test')

def runKBA(X, y, model, output_csv='./results/kba_results.csv'):
    """
    Loads an SVM model from a pickle file, uses the DataLoader to obtain
    feature data, and then makes predictions saving the results to CSV.
    
    Args:
        X: features.
        y: labels
        model_pickle_path (str): Path to the pickle file containing the trained SVM.
        output_csv (str): Path to save the predictions.
    """
    with open(model, 'rb') as f:
        kba_model = pickle.load(f)

    y_pred              = kba_model.predict(X)
    probabilities       = kba_model.predict_proba(X)
    prob_positive_class = probabilities[:, 1]

    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': y_pred,
        'Prediction_Probability': prob_positive_class
    })
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to '{output_csv}'.")

