class Tokenizer :
    def __init__(self):
        
        
        vocab = ['<PAD>', '<SOS>', '<EOS>', 'x', '+', '-', '^', 
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        self.token2id={t:i for i,t in enumerate(vocab)}
        self.id2token={i:t for i,t in enumerate(vocab)}

        self.PAD=self.token2id['<PAD>']
        self.SOS=self.token2id['<SOS>']
        self.EOS=self.token2id['<EOS>']
        self.vocab_size=len(vocab)

    def encode(self,string): #convert each unique chsrcarcter into token,#character level tokenisation
        return [self.token2id[c] for c in string]
    
    def decode (self,ids): #convert each tokensto characters and rebuild the string 
        return ''.join(self.id2token[i] for i in ids)