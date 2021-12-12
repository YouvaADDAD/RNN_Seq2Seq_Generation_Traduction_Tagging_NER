
from torch.optim import optimizer
from utils import RNN, device,ForecastMetroDataset
from torch.utils.data import DataLoader 
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
#  TODO:  Question 3 : Prédiction de séries temporelles


if __name__=='__main__':
    # Nombre de stations utilisé
    CLASSES = 10
    #Longueur des séquences
    LENGTH = 20
    # Dimension de l'entrée (1 (in) ou 2 (in/out))
    DIM_INPUT = 2
    #Taille du batch
    BATCH_SIZE = 512

    #Le forcast
    HORRIZON=3
    #Tensorboard
    writer = SummaryWriter(f"runs/Forecast_{CLASSES}_{LENGTH}_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #Creation des dataSet train et test
    d_train,d_test=torch.load('/Users/addadyouva/Downloads/AMAL/TME4/data/hzdataset.pch') 
    
    #Traitement
    ds_train = ForecastMetroDataset(d_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
    ds_test = ForecastMetroDataset(d_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
    #Dataloader, pas de shuffle sur test pas la peine
    data_train=DataLoader(ds_train,shuffle=True,batch_size=BATCH_SIZE)
    data_test=DataLoader(ds_test,batch_size=len(ds_test))
    
    #Parameters
     #Parameters
    HIDDEN_SIZE=200
    EPS=1e-3
    NB_EPOCH=1000



    #Creation du model
    model=RNN(DIM_INPUT,DIM_INPUT,hidden_size=HIDDEN_SIZE).to(device=device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=EPS)
    loss_fn= torch.nn.L1Loss()
    for i in range(NB_EPOCH):
        #accumulate=[]
        for x,y in data_train:
            batch=x.shape[0]*x.shape[2]
            x=x.permute(1,0,2,3).flatten(start_dim=1,end_dim=2)
            y=y.permute(1,0,2,3).flatten(start_dim=1,end_dim=2)
            h = torch.zeros(batch,HIDDEN_SIZE,requires_grad=True)
            ht,hn=model(x,h)
            yt=torch.stack([model.decode(h)  for h in ht],dim=0)
            loss=loss_fn(yt,y)
            print(f'Loss {loss} a epoche {i}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #accumulate.append(loss)
            writer.add_scalar('Loss/train', loss, i)
            #Forecast
            h_T_1=model.one_step(yt[-1],hn)
            predicted=model.decode(h_T_1)
            
        
        
        #Test and Forecast
        with torch.no_grad():
            x,y =next(iter(data_test))
            length=len(x)
            batch=x.shape[0]*x.shape[2]
            x=x.permute(1,0,2,3).flatten(start_dim=1,end_dim=2)
            y=y.permute(1,0,2,3).flatten(start_dim=1,end_dim=2)
            h = torch.zeros(batch,HIDDEN_SIZE,requires_grad=True)
            ht,hn=model(x,h)
            yt=torch.stack([model.decode(h)  for h in ht],dim=0)
            loss=loss_fn(yt,y)
            print(f'Loss {loss} a epoche {i}')
            writer.add_scalar('Loss/test', loss, i)

            #Forecast sur le test
            forcasted=[]
            h_T_1=hn
            predicted=yt[-1]
            for t in range(HORRIZON):
                h_T_1=model.one_step(predicted,h_T_1)
                predicted=model.decode(h_T_1)
                forcasted.append(predicted)
            #Forcasted
            predicted=torch.stack(forcasted).reshape(HORRIZON,-1,CLASSES,DIM_INPUT)




"""
    #Creation du model
    model=RNN(DIM_INPUT,DIM_INPUT,hidden_size=HIDDEN_SIZE).to(device=device)
    optimizer=torch.optim.Adam(model.parameters(), lr=EPS)
    loss_fn= torch.nn.L1Loss()
    for i in range(NB_EPOCH):
        accumulate=[]
        for x,y in data_train:
            loss=0
            for s in range(CLASSES):
                batch=len(x)
                input=x[:,:,s,:].permute(1,0,2)
                target=y[:,:,s,:].permute(1,0,2)
                h = torch.zeros(batch,HIDDEN_SIZE,requires_grad=True)
                ht,hn=model(input,h)
                yt=torch.stack([model.decode(h)  for h in ht],dim=0)
                loss+=loss_fn(yt,target)
            print(f'Loss {loss/CLASSES} a epoche {i}') 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulate.append(loss)

"""