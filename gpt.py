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

block_size = 8 
train_data [:block_size+1]
print(train_data)

x = train_data[:block_size]
y= train_data[1:block_size+1]
for t in range(block_size ):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} then the output is : {target}")

torch.manual_seed(1337)              
batch_size  = 4 # how many parallel sequneces will we process altogether at once
block_size = 8  # what is the maximum context length for predictions 

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch('train')
print("inputs : ")
print(xb.shape)
print(xb)
print("targets :")
      
print(yb.shape)
print(yb)

for b in range (batch_size) : # batch dimension
    for t in range (block_size): #block dimension
        context = xb[b, :t+1]
        target = yb [b,t]
        print(f"when input is {context.tolist()} the target is {target}")


