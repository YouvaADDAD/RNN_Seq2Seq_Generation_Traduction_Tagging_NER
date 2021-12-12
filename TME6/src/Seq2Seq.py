import torch
import torch.nn as nn
from torch.nn.modules import dropout
from  torch.nn.utils.rnn import pack_padded_sequence
from  torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0
EOS = 1
SOS = 2
OOVID = 3

class POS_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, label_number,hidden_size=100,PAD=0,bidirectional=False,dropout=0.5):
        super(POS_LSTM,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size=embed_size
        self.label_number=label_number
        self.hidden_size = hidden_size
        self.dropout=nn.Dropout(dropout)
        self.embedding=nn.Embedding(vocab_size,embed_size,padding_idx=PAD)
        self.lstm=nn.LSTM(embed_size,hidden_size,bidirectional=bidirectional)
        self.decoder=nn.Linear(hidden_size * 2 if bidirectional else hidden_size, label_number)

    def forward(self, x,lengths):
        output = self.embedding(x)
        embedded = pack_padded_sequence(output,lengths.cpu(),enforce_sorted=False)  
        output,_ = self.lstm(embedded)
        output,_  = torch.nn.utils.rnn.pad_packed_sequence(output) 
        result = output
        result = self.dropout(result)
        out = self.decoder(result)
        return out

class Encoder(nn.Module):
    #encodeur: un embedding du vocabulaire d'origine puis un gru
    def __init__(self, vocab_origine,emb_size, hidden_size):
        super(Encoder,self).__init__()
        self.vocab_origine=vocab_origine
        self.hidden_size=hidden_size
        self.emb_size=emb_size
        self.embedding = nn.Embedding(vocab_origine, emb_size,padding_idx=PAD)
        self.gru = nn.GRU(emb_size, hidden_size)

    def forward(self, input,lengths,hidden=None):
        embedded = self.embedding(input) #Input pad_sequence
        embedded = pack_padded_sequence(embedded,lengths.cpu(),enforce_sorted=False)                                                                                                       
        outputs, hidden = self.gru(embedded,hidden)
        output,_  = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        return output, hidden

class Decoder(nn.Module):

    #décodeur : un embedding du vocabulaire de destination, puis un GRU 
    # suivi d'un réseau linéaire pour le décodage de l'état latent (et un softmax pour terminer)

    def __init__(self,vocab_destination,emb_size,hidden_size,bidirectional=False):
        super(Decoder,self).__init__()
        self.vocab_destination=vocab_destination
        self.hidden_size = hidden_size
        self.emb_size=emb_size
        self.embedding = nn.Embedding(vocab_destination, emb_size,padding_idx=PAD)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.decoder = nn.Linear(hidden_size  , vocab_destination)
    
    def forward(self,input,lengths,hidden):
        #For teacher forcing
        """
        input: Length x batch 
        hidden: tensor of shape (1, batch, hidden_size)
        """
        target = torch.tensor([[SOS]*input.shape[1]],device=device) # 1 x Batch
        target=self.embedding(target)#1 x Batch x emb_size 
        _, decoder_hidden = self.gru(target, hidden)  #1 x batch x hidden_size , 1 x batch x hidden_size
        target=self.embedding(input)# Length x batch x emb_size
        target = pack_padded_sequence(target,lengths.cpu(),enforce_sorted=False)        
        outputs,hidden=self.gru(target,decoder_hidden)
        output,_  = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        return F.log_softmax(self.decoder(output),dim=-1)    #(Length x Batch)

    def generate(self,hidden,lenseq=None):
        #For constraint mode
        """
        lenseq:length
        hidden:1 x Batch x hidden_size
        """
        batch_size=hidden.shape[1]
        decoder_outputs = torch.full(size=[lenseq, batch_size, self.vocab_destination], fill_value=PAD,dtype=torch.float,device=device)
        
        for i in range(batch_size):
            target = torch.tensor([[SOS]],device=device)
            hidden_example=hidden[:,i,:].unsqueeze(1) #1 x 1 x hidden_size
            
            for j in range(lenseq):
                target = self.embedding(target).view(1, 1, -1) # one example and one word -> 1 x 1 x emb_size
                output, hidden_example = self.gru(target, hidden_example)
                decoder_output = F.log_softmax(self.decoder(hidden_example), dim=-1).squeeze()# vocab_size
                indice_max = decoder_output.argmax() #1
                target = indice_max.detach() #1
                decoder_outputs[j,i]=decoder_output
                
                if target.item() == EOS:
                    break
        return decoder_outputs

       

    





