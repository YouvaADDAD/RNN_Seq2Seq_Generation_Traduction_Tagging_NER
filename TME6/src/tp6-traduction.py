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
import datetime
from pathlib import Path
from typing import List
from itertools import chain
import time
import re
from torch.utils.tensorboard import SummaryWriter
from Seq2Seq import * 



def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len




if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    FILE = "./data/en-fra.txt"
    ############################################################################################
    with open(FILE) as f:
        lines = f.readlines()
    ############################################################################################
    lines = [lines[x] for x in torch.randperm(len(lines))]
    idxTrain = int(0.8*len(lines))
    ############################################################################################
    vocEng = Vocabulary(True)
    vocFra = Vocabulary(True)
    ############################################################################################
    MAX_LEN=3
    BATCH_SIZE=64
    ############################################################################################
    datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
    datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)
    ############################################################################################
    train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
    ############################################################################################
    VOCAB_ORIGINE=len(vocEng)
    VOCAB_DESTINATION=len(vocFra)
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
    writer = SummaryWriter('runs/Traduction'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
        writer.add_scalar('Loss/train',loss,i)

        #Test
        with torch.no_grad():
            acc=0
            for orig , o_len, dest ,d_len in test_loader:
                enc_output, enc_hidden = encoder(orig.to(device), o_len.to(device))
                dec_output=decoder.generate(enc_hidden,lenseq=max(d_len))
                loss = loss_fn(dec_output.permute(0,2,1), dest.to(device))
                acc+=loss
            print(f'Loss/Test {acc/len(train_loader)} a l"epoche {i}')
            writer.add_scalar('Loss/Test',loss,i)
    
    #Traduction Example 
    test_loader = DataLoader(datatest, collate_fn=collate, batch_size=1, shuffle=True)
    for orig , o_len, dest ,d_len in test_loader:
        enc_output, enc_hidden = encoder(orig.to(device), o_len.to(device))
        dec_output=decoder.generate(enc_hidden,lenseq=max(d_len)).argmax(dim=-1)
        print('Original word=>'," ".join(vocEng.getwords(orig)))
        print('Traduction>'," ".join(vocFra.getwords(dec_output)))
    
    """
    DECODER OK
    orig,o_len,dest,d_len = next(iter(train_loader))
    enc_output, enc_hidden = encoder(orig, o_len)
    output,dec_output,dec_hidden =decoder(dest,d_len,enc_hidden)
    print(VOCAB_DESTINATION)
    print(output.size())
    print(dec_output.size())
    print(dec_hidden.size())
   
    
    
    ENCODER OK
    enc_output, enc_hidden = encoder(orig, o_len)
    print(max(o_len))
    print(enc_output.size())
    print(enc_hidden.size())

    orig,o_len,dest,d_len = next(iter(train_loader))
    enc_output, enc_hidden = encoder(orig, o_len)
    print('encoder Hidden',enc_hidden.shape )
    output1=decoder.generate(enc_hidden,lenseq=max(d_len))
    output2=decoder(dest,d_len,enc_hidden)
    print('output1',output1.shape)
    print('output2',output2.shape)
    print(loss_fn(output1.permute(0,2,1),dest))
    
    """
    
    
#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
