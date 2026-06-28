# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters : " , len(text))    
print(text[:1000])

#here are all the unique characters in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#creating a mapping and some lookuptables and econding and decoding 
#from strings to integers to integers to strings

stoi = { ch:i for i ,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate (chars)}
encode = lambda s : [stoi[c] for c in s ] # just converted string to a list of integers # enocder
decode = lambda l : ''.join ([itos[i] for i in l])# just converted a list  of integers to a string # decoder

print( encode("ayush pakhale"))
print(decode(encode("ayush pakhale")))
# let me commit someemthinh

#lts not enode the entire datset and store it into torch.tensor
import torch 
data = torch.tensor(encode(text) , dtype = torch.long)
print(data.shape , data.dtype)
print( data [:1000]) # the 1000 characters we looked at earlier will go to the gpt like this 

#lets now split the data into two sets : training and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
