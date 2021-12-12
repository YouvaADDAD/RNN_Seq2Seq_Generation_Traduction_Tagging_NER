
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import datetime
import textloader

#  TODO: 

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
        return t

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    loss_fn=nn.NLLLoss(reduction='mean')
    output=output.flatten(end_dim=1)
    target=target.flatten()
    mask=torch.ne(target,padcar)
    masked_y = torch.masked_select(target, mask)
    masked_out=output[mask]
    return loss_fn(masked_out,masked_y)


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self,input_size,output_size,hidden_size=5,act=nn.Tanh(),final_activation=True):
        """
        x est length×batch×dim 
        h de taille batch×latent
        """
        super(RNN,self).__init__()
        self.Wx=nn.Linear(input_size,hidden_size)
        self.Wh=nn.Linear(hidden_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,output_size)
        self.act=act
        self.final_activation=final_activation
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size


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
        for xt in range(x.size(0)):
            h=self.one_step(x[xt],h) 
            hAll.append(h)
        return torch.stack(hAll,dim=0),h
    
    def decode(self,h):
        """
        h: batch × latent 
        return: batch × output.
        """
        #return self.decoder(h)
        if self.final_activation:
            return F.log_softmax(self.decoder(h),dim=-1)  #-> On enleve pour faire du forecasting
        return self.decoder(h)


class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self,input_size,output_size,hidden_size=5,final_activation=True):
        super(LSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.final_activation=final_activation
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.Wh=nn.Linear(input_size+hidden_size,hidden_size)
        self.Wx=nn.Linear(input_size+hidden_size,hidden_size)
        self.Wc=nn.Linear(input_size+hidden_size,hidden_size)
        self.Wo=nn.Linear(input_size+hidden_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,output_size)

        
    
    
    def one_step(self, x, h,c):
        """
        x: batch × dim
        h: batch × latent
        c:batch x latent
        return batch × latent
        """
   
        gate=torch.cat((h,x),dim=1)
        ft=self.sigmoid(self.Wh(gate)) # batch x (latent+input)
        it=self.sigmoid(self.Wx(gate)) # batch x latent 
        Ct=torch.add(torch.mul(ft,c),torch.mul(it,self.tanh(self.Wc(gate)))) #batch x latent 
        ot=self.sigmoid(self.Wo(gate)) # batch x latent 
        ht=torch.mul(ot,self.tanh(Ct)) # batch x latent
        return ht,Ct
    
    def forward(self, x, h,c):
        """
        x:length×batch×dim 
        h:batch×latent
        c:batch×latent
        return:length×batch×latent
        """
        hAll=[]
        for xt in range(x.size(0)):
            h,c=self.one_step(x[xt],h,c)
            hAll.append(h) 
        return torch.stack(hAll,dim=0),(h,c)
    
    def decode(self, h):
        if self.final_activation:
            return F.log_softmax(self.decoder(h),dim=-1)
        return self.decoder(h)
        
        


class GRU(nn.Module):

    #  TODO:  Implémenter un GRU
    def __init__(self,input_size,output_size,hidden_size=5,final_activation=True):
        super(GRU,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.final_activation=final_activation
        self.Wz=nn.Linear(input_size+hidden_size,hidden_size)
        self.Wr=nn.Linear(input_size+hidden_size,hidden_size)
        self.W=nn.Linear(input_size+hidden_size,hidden_size)
        self.decoder=nn.Linear(hidden_size,output_size)
    
    def one_step(self,x,h):
        gate=torch.cat((h,x),dim=1)
        zt=self.sigmoid(self.Wz(gate))#batch x latent
        rt=self.sigmoid(self.Wr(gate))# batch x latent
        ht=torch.mul((1-zt),h)+ torch.mul(zt,self.tanh(self.W(torch.cat((torch.mul(rt,h),x),dim=1))))
        return ht
    
    def forward(self,x,h):
        hAll=[]
        for xt in range(x.size(0)):
            h=self.one_step(x[xt],h)
            hAll.append(h) 
        return torch.stack(hAll,dim=0),h
    
    def decode(self,h):
        if self.final_activation:
            return F.log_softmax(self.decoder(h),dim=-1)
        return self.decoder(h)


       
        

   



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
if __name__=='__main__':

    writer = SummaryWriter('runs/trumps-speech_LSTM_Beam_nucleus_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open('../data/trump_full_speech.txt','r') as file:
        text=file.read()
    BATCH_SIZE=64
    gen=['To al', 'He wa', 'Norma', 'Folks', 'When ', '6 mil', 'It wa', 'In 10', 'They ', 'ISIS ','Applaus']
    #LENGTH=100 the best
    LENGTH=50
    HIDDEN_SIZE=200
    IN_SIZE=len(LETTRES)+2
    OUT_SIZE=len(LETTRES)+2
    NB_EPOCH=50
    EMB_SIZE=80
    #NB_EPOCH=100 the best
    #EMB_SIZE=20 the best
    ###########################################################################
    data=DataLoader(TrumpDataset(text,maxlen=LENGTH), collate_fn=pad_collate_fn, batch_size=BATCH_SIZE,shuffle=True)
    embedding=nn.Embedding(IN_SIZE,EMB_SIZE)
    model=LSTM(EMB_SIZE,OUT_SIZE,hidden_size=HIDDEN_SIZE,final_activation=True)
    decoder=model.decode
    ########################################################################### 
    loss_fn = maskedCrossEntropy
    optim = torch.optim.AdamW(model.parameters(), lr = 0.001)
    ###########################################################################
    for i in range(NB_EPOCH):
        for x,y in data: #x->Length x batch
            batch=x.shape[1]
            x=embedding(x) # Length x BatchX Emb_size
            h0 = torch.randn(batch, HIDDEN_SIZE,requires_grad=True)
            c0 = torch.randn(batch, HIDDEN_SIZE,requires_grad=True)
            if (type(model).__name__=='LSTM'):
                ht,(hn,cn)=model(x,h0,c0)
            else:
                ht,hn=model(x,h0)
            yt=torch.stack([model.decode(h) for h in ht],dim=0) #Lenght x Batch x Output
            loss=loss_fn(yt,y,padcar=PAD_IX)
            print(f'loss {loss} a epoch {i}')
            optim.zero_grad()
            loss.backward()
            optim.step()
            for name, param in model.named_parameters():
                writer.add_histogram(name,param.grad.data, i)
            writer.add_scalar('Loss/generation',loss,i)
    
    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            "embedding" : embedding.state_dict()
            }, 'model.pt')

    for phrase in gen:
        print('#'*20)
        #print(generate(model, embedding, model.decoder, start=phrase,eos=EOS_IX))
        print('with nucleus=>',generate_beam(model, embedding, decoder, eos=EOS_IX, k=5, start=phrase,nucleus=True, maxlen=80))
        print('without nucleus=>',generate_beam(model, embedding, decoder, eos=EOS_IX, k=5, start=phrase,nucleus=False, maxlen=80))
        print('#'*20)
       
    
    
    

