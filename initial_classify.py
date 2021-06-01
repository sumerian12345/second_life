import argh
# import clip
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
from CLIP.clip import clip
from CLIP.clip import model
import sys
from pathlib import Path
# from clip import model

# clip_dir = Path(".").absolute() / "CLIP"

# sys.path.append(str(clip_dir))

# print(f"CLIP dir is: {clip_dir}")

def clip_classify(folder,element,img):
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    
    # clip_dir = Path(".").absolute() / "CLIP"
    # sys.path.append(str(clip_dir))

    if element == "door":
        class_names = ["a modern door","an old fashioned door","white door", "brown door", "flush door", "panelled door", "louvered door", "ledged door","a door in a good condition", "a door in a bad condition", "a door with and opening"]
        
        clip_dir = Path(".").absolute() / "CLIP"

        sys.path.append(str(clip_dir))

        with open(r'C:\\Users\\Admin\\Google Drive\\AIA Studio\\second_life\\model1.pickle', 'rb') as handle:
            model = pickle.load(handle)
    elif element == "window":
        #FILL WITH CLASSES AND LOCATION OF PKL FILE
        class_names = ["new","old"]
    elif element == "lumber":
        #FILL WITH CLASSES AND LOCATION OF PKL FILE
        class_names = ["new","old"]
    elif element == "flooring":
        #FILL WITH CLASSES AND LOCATION OF PKL FILE
        class_names = ["new","old"]
    else:
        class_names = ["new","old"]
    

    class_captions = [f"An image depicting a {x}" for x in class_names]
    #print(class_captions)
    
    text_input = clip.tokenize(class_captions).to(device)
    #print(f"Tokens shape: {text_input.shape}")

    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    #print(f"Text features shape: {text_features.shape}")

    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        image *= image_std[:, None, None]    
        image += image_mean[:, None, None]
        return image

    imgFol= folder.split("/")
    fld = ""
    for p in range(len(imgFol)-1):
        fld += imgFol[p]+"/"
    fld = fld[:-1]
    #print(fld)

    dataset = ImageFolder(root=fld, transform=transform)
    data_batches = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # read out all images and true labels
    image_input, y_true = next(iter(data_batches))
    image_input = image_input.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()


    #show_results(image_features, text_features, class_names):
    # depends on global var dataset

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    k = np.min([len(class_names), 5])
    # top_probs, top_labels = text_probs.cpu().topk(k, dim=-1)
    text_probs = text_probs.cpu()

    #plt.figure(figsize=(26, 16))

    probs_lst = []
    probs_recommend = []
    for i in range(len(class_names)):
        #print(class_names[i])
        probs_lst.append(text_probs[img][i].item())
        if text_probs[img][i].item() >0.05:
            probs_recommend.append(class_names[i])

    return probs_lst , probs_recommend

def clip_clasify_element(folder):
    
    #folder = '/content/gdrive/MyDrive/StudioAI/CLIP_ELEM_CLASS/'
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    
    initial_class_names = ["door","window","flooring","lumber"]
    class_captions = [f"An image depicting a {x}" for x in initial_class_names]
    text_input = clip.tokenize(class_captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    #print(f"Text features shape: {text_features.shape}")
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

    # if os.path.isfile(folder+"/"+".csv"):
    #     database = pd.read_csv(folder+"/"+".csv")  
    #     print("Database exists")
    # else:
    #     database = pd.DataFrame(files, columns =['Images'])
    #     print("Create new database")
    
    for i, (image, label_idx) in enumerate(dataset):
        max_prob = 0 
        element = []
        idx = ""
        for j in range(len(initial_class_names)):
            if text_probs[i][j].item() > max_prob:
                max_prob = text_probs[i][j].item()
                idx = j
        element = initial_class_names[idx]
    
        #print("It's ",element)
        
        probs_data = clip_classify(folder,element,i)[0]
        probs_recom = clip_classify(folder,element,i)[1]
        print("It's ",element, " ",probs_recom)
    
        #database[class_names[i]] = pd.Series(probs_data, index=database.index)
        
    #database.to_csv(r''+folder+"/"+".csv", index = False)
    
    



# assembling:

parser = argh.ArghParser()
parser.add_commands([clip_clasify_element])


if __name__ == '__main__':
    print("Starting the classification file")
    parser.dispatch()
    