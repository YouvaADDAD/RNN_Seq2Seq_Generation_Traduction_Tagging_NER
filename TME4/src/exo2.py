from torch.optim import optimizer
from utils import RNN, device,SampleMetroDataset
from torch.utils.data import DataLoader 
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

if __name__=='__main__':

    # Nombre de stations utilisé
    CLASSES = 80
    #Longueur des séquences
    LENGTH = 60
    # Dimension de l'entrée (1 (in) ou 2 (in/out))
    DIM_INPUT = 2
    #Taille du batch
    BATCH_SIZE = 128

    writer = SummaryWriter(f"runs/runs-Classification_{CLASSES}_{LENGTH}_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #Creation des dataSet train et test
    d_train,d_test=torch.load('/Users/addadyouva/Downloads/AMAL/TME4/data/hzdataset.pch') 
    
    #Traitement
    ds_train=SampleMetroDataset(d_train[:,:,:CLASSES,:DIM_INPUT],length=LENGTH)
    ds_test=SampleMetroDataset(d_test[:,:,:CLASSES,:DIM_INPUT],length=LENGTH,stations_max=ds_train.stations_max)

    #Dataloader, pas de shuffle sur test pas la peine
    data_train=DataLoader(ds_train,shuffle=True,batch_size=BATCH_SIZE)
    data_test=DataLoader(ds_test,batch_size=len(ds_test))
    
    #Parameters
     #Parameters
    HIDDEN_SIZE=200
    EPS=1e-3
    NB_EPOCH=1800


    #Creation du model
    model=RNN(DIM_INPUT,CLASSES,hidden_size=HIDDEN_SIZE).to(device=device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=EPS)
    loss_fn= torch.nn.NLLLoss()

    for i in range(NB_EPOCH):
        for x,y in data_train:
            
            batch=len(x)
            x=x.permute(1,0,2)
            h = torch.zeros(batch,HIDDEN_SIZE,requires_grad=True)
            ht,hn = model(x,h)
            decode = model.decode(hn)
            loss=loss_fn(decode,y) #NLLLoss -> prend input taille (N,C) ou C est le nombre de classe
            loss.backward()
            print(f'loss Learn->{loss}')
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss/train', loss, i)
            
        with torch.no_grad():
            for x,y in data_test:
                batch=len(x)
                x=x.permute(1,0,2)
                h = torch.zeros(batch,HIDDEN_SIZE,requires_grad=True)
                ht,hn = model(x,h)
                decode = model.decode(hn)
                accuracy = torch.sum((decode.argmax(1) == y)) / y.shape[0] 
                loss=loss_fn(decode,y) #NLLLoss -> prend input taille (N,C) ou C est le nombre de classe
                writer.add_scalar('Loss/test', loss, i)
                writer.add_scalar('Loss/Accuracy', accuracy, i)
                print(f'loss Test->{loss}')


            