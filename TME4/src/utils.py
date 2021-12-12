import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self,input_size,output_size,hidden_size=5,act=nn.Tanh(),final_activation=nn.LogSoftmax(dim=-1)):
        """
        x est length×batch×dim 
        h de taille batch×latent
        """
        super(RNN,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.Wx=nn.Linear(input_size,hidden_size)
        self.Wh=nn.Linear(hidden_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,output_size)
        self.act=act
        self.final_activation=final_activation


    def one_step(self,x,h):
        """
        x: batch × dim
        h: batch × latent
        return batch × latent
        """
        return self.act(torch.add(self.Wx(x),self.Wh(h)))

    def forward(self,x,h):
        """
        x:length×batch×dim 
        h:batch×latent.
        return:length×batch×latent
        """
        hAll=[]
        for xt in x:
            h=self.one_step(xt,h) 
            hAll.append(h)
        return torch.stack(hAll,dim=0),h
    
    def decode(self,h):
        """
        h: batch × latent 
        return: batch × output.
        """
        #return self.decoder(h)
        if self.final_activation:
            return self.final_activation(self.decoder(h))  #-> On enleve pour faire du forecasting
        return self.decoder(h)
            

        

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        self.stations_max=stations_max
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        self.stations_max=stations_max
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]
