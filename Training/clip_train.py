import argh

def clip-train(folder, label_csv, lr= 0.01, epoch= 50, batch =32):
    
    import pandas as pd

    #Location of the folder containing images and CSV file
    location = folder
    
    #Open CSV file with a dataset containing image name and labels
    waste_df = label_csv

    #Convert image name into path of the image
    waste_df["Image"] = location+waste_df["Image"]
    
    #Convert the label in the dataset into text
    waste_df["Condition"] = "An image of a door in a "+waste_df["Condition"]+" condition"
    
    #Remove all the columns that are not used for the traning
    waste_df.drop("Element",inplace=True, axis=1)
    waste_df.drop("Opening",inplace=True, axis=1)
    waste_df.drop("Color",inplace=True, axis=1)
    
    #Look at the DF
    print(waste_df.shape)
    waste_df
    
    # In order to display the image we will need to de-nonrmalize them
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to('cpu')
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to('cpu')
    
    def denormalize_image(image: torch.Tensor) -> torch.Tensor:
        image *= image_std[:, None, None]    
        image += image_mean[:, None, None]
        return image

    X = waste_df["Image"]
    y = waste_df["Condition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16


    #SET LOSS
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    #SET OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #Other possibilities
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper

    #DATASET LOADER

    #import class Dataset and Dataloader fompyTorch
    from torch.utils.data import Dataset, DataLoader

    #SHOW RESULTS - image with probability bar chart
    def show_results(logits_per_image, class_names, images, label_idxs):

        # depends on global var dataset
        text_probs = logits_per_image
        k = np.min([len(class_names), 5])
    
        plt.figure(figsize=(26, 16))
    
        for i, (image, label_idx) in enumerate(zip(images, label_idxs)):
            plt.subplot(4, 8, 2 * i + 1)
            plt.imshow(denormalize_image(image).permute(1, 2, 0))
            plt.axis("off")
    
            plt.subplot(4, 8, 2 * i + 2)
            y = np.arange(k)
            plt.grid()
            plt.barh(y, text_probs[i])
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.yticks(y, class_names)
            plt.xlabel("probability")
    
        plt.subplots_adjust(wspace=0.5)
        plt.show() 

    # Convert model - based on https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    # CLASS TO PRE-PROCESS IMAGE using CLIP
    class image_caption_dataset(Dataset):
        def __init__(self, X,y):
    
            self.images = X.tolist()
            self.caption = y.tolist()
    
        def __len__(self):
            return len(self.caption)
    
        def __getitem__(self, idx):
            images = preprocess(Image.open(self.images[idx])) #preprocess from clip.load
            caption = self.caption[idx]
            return images,caption

    #CREATE DATASET WITH PRE-PROCESSED IMAGES

    dataset = image_caption_dataset(X_train, y_train)
    dataset_test = image_caption_dataset(X_test, y_test)

    # SET TRAINING PARAMETERS

    EPOCH = epoch
    BATCH_SIZE = batch

    train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader
    test_dataloader = DataLoader(dataset_test,batch_size = 1) #Define your own dataloader

    #DEFINE CLASS NAMES 
    class_names = list(set(y_train))

    loss_plt = []
    accuracy_plt=[]
    accuracy_train=[]

    for epoch in range(EPOCH):

        #TESTING
        print("Testing")
    
        text_input = clip.tokenize(class_names).to(device)
        correct = 0
        N = 0
        logits_per_images = []
        images = []
        label_idxs = []
    
        for batch in test_dataloader :
          list_image = batch[0].to(device)
          true_text = batch[1]
          logits_per_image, logits_per_text = model(list_image, text_input)
          predicted_text = class_names[logits_per_image[0].argmax()]
          N += 1
          correct += predicted_text == true_text[0]
          logits_per_images.append(logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0])
          images.append(list_image[0].detach().cpu())
          label_idxs.append(class_names.index(true_text[0]))
    
        print("Test accuracy " + str(correct/N*100))
        accuracy_plt.append(correct/N*100)
        show_results(logits_per_images[0:16], class_names, images[0:16], label_idxs[0:16]) 
    
        #TRAIN THE MODEL
        losses = []
        for batch in train_dataloader:
    
            if len(batch[0]) != BATCH_SIZE:
                print("Skipping")
                continue
        
            optimizer.zero_grad()
        
            list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
        
            images = list_image.to(device)
            texts = clip.tokenize(list_txt).to(device)
        
            logits_per_image, logits_per_text = model(images, texts)
        
            ground_truth = torch.arange(BATCH_SIZE).long().to(device)
        
            total_loss = (loss_img(logits_per_image,ground_truth) + 1.0*loss_txt(logits_per_text,ground_truth))/(BATCH_SIZE * 2)
        
            total_loss.backward()
            losses.append(total_loss.item())
        
            print(total_loss)
        
            if device == "cpu":
                #modify parameters to reduce total loss
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
    
        N, correct = 0, 0
        for batch in train_dataloader :
            list_image = batch[0].to(device)
            true_text = batch[1]
            logits_per_image, logits_per_text = model(list_image, text_input)
        
            predicted_text = class_names[logits_per_image[0].argmax()]
            N += 1
            correct += predicted_text == true_text[0]
            if N > len(test_dataloader):
            break
        
        print("Train accuracy " + str(correct/N*100))
        accuracy_train.append(correct/N*100)  
    
        loss_plt.append(np.mean(losses))
        print("Finished " + str(epoch) + "epoch. Mean loss:")
        print(np.mean(losses))


# assembling:

parser = argh.ArghParser()
parser.add_commands([clip-train])


if __name__ == '__main__':
    parser.dispatch()