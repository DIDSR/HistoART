import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import timm
import csv
from   tqdm import tqdm

def save_predictions(epoch, phase, batch_idx, inputs, labels, predicted, probs, output_file):
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
    uni_model = setup_uni_model()
    uni_model.load_state_dict(torch.load(model, map_location=device))
    uni_model.to(device)
    uni_model.eval()

    test_loop(uni_model, dataloader, 0, device, './results/fma_results.csv', 'test')

def runDLA(dataloader, device, model):
    resnet_model = setup_resnet_model()
    resnet_model.load_state_dict(torch.load(model, map_location=device))
    resnet_model.to(device)
    resnet_model.eval()

    test_loop(resnet_model, dataloader, 0, device, './results/dla_results.csv', 'test')

## def runKBA():

