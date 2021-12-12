import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import datetime
from utils import RNN, device
import torch.nn.functional as F
from torch.nn.parameter import *

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]




#  TODO: 

if __name__=='__main__':

    writer = SummaryWriter('runs/trumps-speech'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open('../data/trump_full_speech.txt','r') as file:
        text=file.read()
    BATCH_SIZE=64
    LENGTH=100
    HIDDEN_SIZE=100
    IN_SIZE=len(LETTRES)+1
    OUT_SIZE=len(LETTRES)+1
    NB_EPOCH=50
    EMB_SIZE=20
    ###########################################################################
    data=DataLoader(TrumpDataset(text,maxlen=LENGTH),batch_size=BATCH_SIZE,shuffle=True)
    #one_hot_encoding=torch.eye(IN_SIZE)
    #embedding=torch.nn.Linear(IN_SIZE,EMB_SIZE,bias=False)
    embedding = Parameter(torch.empty(IN_SIZE, EMB_SIZE).normal_())
    model=RNN(EMB_SIZE,OUT_SIZE,hidden_size=HIDDEN_SIZE,final_activation=torch.nn.LogSoftmax(dim=-1))
    ###########################################################################
    loss_fn = torch.nn.NLLLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = 0.001)
    ###########################################################################
    for i in range(NB_EPOCH):
        for x,y in data:
            x=x.transpose(1,0)
            y=y.transpose(1,0)
            size=x.size()
            #x=embedding(one_hot_encoding[x])
            x=embedding.index_select(0, x.reshape(-1)).view(size[0],size[1],-1)
            h0 = torch.randn(size[1], HIDDEN_SIZE,requires_grad=True)
            ht,hn=model(x,h0)
            yt=torch.stack([model.decode(h)  for h in ht],dim=0)
            loss=loss_fn(yt.permute(0,2,1),y)
            print(f'loss {loss} a epoch {i}')
            optim.zero_grad()
            loss.backward()
            optim.step()
            writer.add_scalar('Loss/generation',loss,i)
            
    
    first_char=''
    predicted=first_char
    input=torch.tensor([lettre2id['']])
    size=input.size()
    h = torch.zeros(1,HIDDEN_SIZE,requires_grad=True)
    input=embedding.index_select(0, input.reshape(-1)).view(size[0],-1)
    LENGTH=40
    for p in range(LENGTH):
        h=model.one_step(input,h)
        char_pred=torch.exp(model.decode(h))
        best_i = torch.multinomial(char_pred, 1)[0]
        predicted+=code2string(best_i)
        input=embedding.index_select(0, best_i).view(1,-1)
    print(predicted)
    


    

   

