import torch
import torch.nn
import torch.nn.functional as F
import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
from itertools import chain
import datetime
import time
import re
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from  torch.nn.utils.rnn import pack_padded_sequence
from  torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import sentencepiece as sp
from Seq2Seq import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################################################################ 
#create our English SentencePiece model
sp.SentencePieceTrainer.Train(input="./data/en.txt",model_prefix='./data/en',vocab_size=7000)
en = sp.SentencePieceProcessor(model_file='./data/en.model')

#create our French SentencePiece model
sp.SentencePieceTrainer.Train(input="./data/fr.txt",model_prefix='./data/fr',vocab_size=7000)
fr = sp.SentencePieceProcessor(model_file='./data/fr.model')

PAD = en.pad_id()
EOS = en.eos_id()
SOS = en.bos_id()


def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()

def fic_splited():
    #########Load File#####################
    FILE = "./data/en-fra.txt"
    ############################################################################################
    with open(FILE) as f:
        lines = f.readlines()
    original=[]
    destination=[]
    for line in tqdm(lines):
        for s in line.split("\n"):
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)<1:continue
            original.append(orig)
            destination.append(dest)

    with open('./data/en.txt', 'w') as f:
            for ori in original:
                f.write(ori + "\n")

    with open('./data/fr.txt', 'w') as f:
            for dest in destination:
                f.write(dest + "\n")


class TradDataset():
    def __init__(self,dataOriginal,dataDestination,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for orig,dist in zip(dataOriginal,dataDestination):
            if len(orig)<1: continue
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor(vocOrig.encode(orig, out_type=int)),torch.tensor(vocDest.encode(dist, out_type=int))))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    FILE_ORIGINAL = "./data/en.txt"
    FILE_DESTINATION = "./data/fr.txt"
    ############################################################################################
    with open(FILE_ORIGINAL) as f:
        dataOriginal = f.readlines()
        
    with open(FILE_DESTINATION) as f:
        dataDestination = f.readlines()
        
    ############################################################################################
    dataOriginal = [dataOriginal[x] for x in torch.randperm(len(dataOriginal))]
    dataDestination = [dataDestination[x] for x in torch.randperm(len(dataDestination))]
    idxTrain = int(0.8*len(dataOriginal))
    ############################################################################################
    MAX_LEN=5
    BATCH_SIZE=64
    ############################################################################################
    datatrain = TradDataset(dataOriginal[:idxTrain],dataDestination[:idxTrain],en,fr,max_len=MAX_LEN)
    datatest = TradDataset(dataOriginal[idxTrain:],dataDestination[idxTrain:],en,fr,max_len=MAX_LEN)
    ############################################################################################
    train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    ############################################################################################
    VOCAB_ORIGINE=en.vocab_size()
    VOCAB_DESTINATION=fr.vocab_size()
    HIDDEN_SIZE=256
    EMB_SIZE=256
    NB_EPOCH=100
    ############################################################################################
    encoder=Encoder(VOCAB_ORIGINE,EMB_SIZE, HIDDEN_SIZE).to(device)
    decoder=Decoder(VOCAB_DESTINATION,EMB_SIZE, HIDDEN_SIZE).to(device)
    ############################################################################################
    loss_fn=nn.NLLLoss(ignore_index=PAD)
    ############################################################################################
    optimizer=optim.RMSprop(chain(encoder.parameters(),decoder.parameters()), lr=0.001)
    #writer = SummaryWriter('runs/Traduction'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ############################################################################################
    tf_prob = 0.5

    for i in range(NB_EPOCH):
        acc=0
        for orig , o_len, dest ,d_len in train_loader:
            enc_output, enc_hidden = encoder(orig.to(device), o_len.to(device))
            tf_use = True if torch.rand(1) < tf_prob else False
            if tf_use:
                dec_output=decoder(dest.to(device),d_len.to(device),enc_hidden) #OK 
            else:
                dec_output=decoder.generate(enc_hidden,lenseq=max(d_len))
            loss = loss_fn(dec_output.permute(0,2,1), dest.to(device))
            acc+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Loss/Train {acc/len(train_loader)} a l"epoche {i}')
        #writer.add_scalar('Loss/train',loss,i)

        #Test
        with torch.no_grad():
            acc=0
            for orig , o_len, dest ,d_len in test_loader:
                enc_output, enc_hidden = encoder(orig.to(device), o_len.to(device))
                dec_output=decoder.generate(enc_hidden,lenseq=max(d_len))
                loss = loss_fn(dec_output.permute(0,2,1), dest.to(device))
                acc+=loss
            print(f'Loss/Test {acc/len(train_loader)} a l"epoche {i}')
            #writer.add_scalar('Loss/Test',loss,i)
    
    #Traduction Example 
    test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    for orig , o_len, dest ,d_len in test_loader:
        enc_output, enc_hidden = encoder(orig.to(device), o_len.to(device))
        dec_output=decoder.generate(enc_hidden,lenseq=max(d_len)).argmax(dim=-1)
        print('Original word=>'," ".join(en.decode(orig.tolist())))
        print('Traduction>'," ".join(fr.decode(dec_output.tolist())))