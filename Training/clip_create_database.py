import argh
import clip
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle


def clip_classify(class_names,folder,element):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)

    #with open('./Door/model1.pickle', 'rb') as handle:
    #    model = pickle.load(handle)

    files = os.listdir(folder)

    class_names=class_names.replace("_", " ")
    class_names= class_names.split(",")
    
    class_captions = [f"An image depicting a {x}" for x in class_names]
    
    text_input = clip.tokenize(class_captions).to(device)
    print(f"Tokens shape: {text_input.shape}")

    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    print(f"Text features shape: {text_features.shape}")

    # In order to display the image we will need to de-nonrmalize them
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to('cpu')
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to('cpu')

    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        image *= image_std[:, None, None]    
        image += image_mean[:, None, None]
        return image

    imgFol= folder.split("/")
    fld = ""
    for p in range(len(imgFol)-1):
        fld += imgFol[p]+"/"
    fld = fld[:-1]
    print(fld)

    dataset = ImageFolder(root=fld, transform=transform)
    data_batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # read out all images and true labels
    image_input, y_true = next(iter(data_batches))
    image_input = image_input.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    k = np.min([len(class_names), 5])

    text_probs = text_probs.cpu()

    if os.path.isfile(folder+"/"+element+".csv"):
        database = pd.read_csv(folder+"/"+element+".csv")  
        print("Database exists")
    else:
        database = pd.DataFrame(files, columns =['Images'])
        print("Create new database")
    
    probs_lst = []
    for i in range(len(class_names)):
        print(class_names[i])
        probs_sub = []
        print(len(dataset))
        for j, (image, label_idx) in enumerate(dataset):
            probs_sub.append(text_probs[j][i].item())

        database[class_names[i]] = pd.Series(probs_sub, index=database.index)

    database.to_csv(r''+folder+"/"+element+".csv", index = False)
    
# assembling:

parser = argh.ArghParser()
parser.add_commands([clip_classify])


if __name__ == '__main__':
    parser.dispatch()