from textloader import  string2code, lettre2id,code2string
import math
import torch
from tp5 import  RNN,LSTM,GRU
import torch.nn.functional as F

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près,
    #  i.e. ce qui vient avant le softmax) des différentes sorties possibles

    input  = emb(torch.tensor([lettre2id["<PAD>"]])).view(1,1,-1)
    hidden = torch.zeros(1,rnn.hidden_size,requires_grad=True)
    C      = torch.zeros(1,rnn.hidden_size,requires_grad=True)
    predicted_seq=''+start
    if start:
        input=emb(string2code(start).view(len(start),-1)).view(len(start), 1, -1)
        if (type(rnn).__name__=='LSTM'):
            _,(hidden,C)=rnn(input,hidden,C)
        else:
            _,hidden=rnn(input,hidden)

    for _ in range(maxlen):
        if (type(rnn).__name__=='LSTM'):
            output, (hidden,C) = rnn(input, hidden,C) #input -> 1,1,1, hidden-> 1 x 32, C->1 x 32
        else:
            output, hidden = rnn(input, hidden)
        output = torch.exp(decoder(hidden))
        #top_i = int(torch.multinomial(output, 1)[0])
        top_i = torch.multinomial(output,1)[0]
        predicted_char = code2string(top_i)
        if top_i == eos:
            break
        predicted_seq+=predicted_char
        input=emb(torch.tensor([top_i]).view(1, -1))
    return predicted_seq
      
def generate_beam(rnn, emb, decoder, eos, k,nucleus=True, start="", maxlen=200,alpha=0.95):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
        value,top_i = torch.topk(output,k,1) #Tensor taille 1xK
    """
    #Calcule du premier Hidden
    if nucleus:
        pnucleus=p_nucleus(decoder=decoder,alpha=alpha)
    input  = emb(torch.tensor([lettre2id[" "]])).view(1,1,-1)
    hidden = torch.zeros(1,rnn.hidden_size,requires_grad=True)
    C      = torch.zeros(1,rnn.hidden_size,requires_grad=True)
    predicted_seq=''+start
    set_sequence=[]
    if start:
        coding=string2code(start)
        input=emb(coding.view(len(start),-1)).view(len(start), 1, -1)
        if (type(rnn).__name__=='LSTM'):
            _,(hidden,C)=rnn(input,hidden,C)
        else:
            _,hidden=rnn(input,hidden)
        set_sequence=coding.view(-1).tolist()
    
    if (type(rnn).__name__=='LSTM'):
        output, (hidden,C) = rnn(input, hidden,C) #input -> 1,1,1, hidden-> 1 x 32, C->1 x 32
    else:
        output, hidden = rnn(input, hidden)
    

    if nucleus:
        output=pnucleus(hidden)
    else:
        output = decoder(hidden).squeeze()
    value,top_i = torch.topk(output,k)
    set_sequence=[(set_sequence + [i.item()] ,v.item()) for i,v in  list(zip(top_i,value))]
    cpt=0
    while (len(set_sequence[0][0]) <= maxlen and eos not in set_sequence[0][0]):
        adder=[]
        for sequence,score in set_sequence:
    
            hidden = torch.zeros(1,rnn.hidden_size)
            C      = torch.zeros(1,rnn.hidden_size)
        
            input = emb(torch.tensor(sequence).view(len(sequence), -1)).view(len(sequence), 1, -1)

            if (type(rnn).__name__=='LSTM'):
               output, (hidden,C) = rnn(input, hidden,C) #input -> 1,1,1, hidden-> 1 x 32, C->1 x 32
            else:
                output, hidden = rnn(input, hidden)
        
            if nucleus:
                output=pnucleus(hidden)
            else:
                output = decoder(hidden).squeeze()
            value,top_i = torch.topk(output,k)
            adder+=[(sequence + [i.item()] ,score+v.item()) for i,v in  list(zip(top_i,value))]

        set_sequence=sorted(adder, key=lambda tup: tup[1],reverse=True)[:k]

    return code2string(set_sequence[0][0])

# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        
        probs=F.softmax(decoder(h).squeeze(),dim=-1)
        sorted, indices = torch.sort(probs,descending=True)
        cum_probs=torch.cumsum(sorted,dim=-1)
        remove_indice = cum_probs > alpha
        remove_indice[1:] = remove_indice[:-1].clone()
        remove_indice[0] = 0
        indices=indices[remove_indice]
        return F.log_softmax(probs,dim=-1)
        
    return compute
