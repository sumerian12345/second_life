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

def clip_classify(folder,element,img):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    
    if element == "door":
        class_names = ["a modern door","an old fashioned door","white door", "brown door", "flush door", "panelled door", "louvered door", "ledged door","a door in a good condition", "a door in a bad condition", "a door with and opening"]
        #with open('./Door/model1.pickle', 'rb') as handle:
        #    model = pickle.load(handle)
    elif element == "window":
        class_names = ["a modern window", "a window with stained glass", "an old fashioned window", "white window", "opaque window", "a window in a good condition", "a window in a bad condition"]
        #with open('./Window/model1.pickle', 'rb') as handle:
        #    model = pickle.load(handle)
    elif element == "lumber":
        #FILL WITH CLASSES AND LOCATION OF PKL FILE
        class_names = ["structural","paneling","walls","furniture","flooring"]
        #with open('./Lumber/model1.pickle', 'rb') as handle:
        #    model = pickle.load(handle)
    elif element == "flooring":
        class_names = class_names = ["ceramic_flooring","wood flooring","stone flooring","flooring in a good condition","flooring in a bad condition","white flooring","grey flooring","brown flooring","mixed color flooring"]
        #with open('./Flooring/model1.pickle', 'rb') as handle:
        #    model = pickle.load(handle)
        
    
    else:
        class_names = ["new","old"]
    
    class_captions = [f"An image depicting a {x}" for x in class_names]

    text_input = clip.tokenize(class_captions).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        image *= image_std[:, None, None]    
        image += image_mean[:, None, None]
        return image

    imgFol= folder.split("/")
    fld = ""
    for p in range(len(imgFol)-1):
        fld += imgFol[p]+"/"
    fld = fld[:-1]

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

    probs_lst = []
    probs_recommend = []
    for i in range(len(class_names)):
        probs_lst.append(text_probs[img][i].item())
        if text_probs[img][i].item() >0.5:
            probs_recommend.append(class_names[i])

    return probs_lst , probs_recommend

def clip_clasify_element(folder):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    
    initial_class_names = ["door","window","flooring","lumber"]
    class_captions = [f"An image depicting a {x}" for x in initial_class_names]
    text_input = clip.tokenize(class_captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    fld= folder

    dataset = ImageFolder(root=fld, transform=transform)
    data_batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    image_input, y_true = next(iter(data_batches))
    image_input = image_input.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    text_probs = text_probs.cpu()
    
    files = sorted(os.listdir(folder))

    for i, (image, label_idx) in enumerate(dataset):
        max_prob = 0 
        element = []
        idx = ""
        for j in range(len(initial_class_names)):
            if text_probs[i][j].item() > max_prob:
                max_prob = text_probs[i][j].item()
                idx = j
        element = initial_class_names[idx]
    
        probs_data = clip_classify(folder,element,i)[0]
        probs_recom = clip_classify(folder,element,i)[1]
        print("It's ",element, " ",probs_recom)
    

# assembling:

parser = argh.ArghParser()
parser.add_commands([clip_clasify_element])


if __name__ == '__main__':
    print("Starting the classification file")
    parser.dispatch()
    