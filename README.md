``` python
pip install wget
```

``` python
import wget
```

``` python
import wget

"""
Importing (and hence utilising) the wget function to pull data from a file.
Which in this case will be the dataset that this Generative Pre-Training Transformer
will rely on.

This is the 'The Origin and Development of the Quantum Theory' by Max Planck, obtained
via the Project Gutenberg website.
"""

# Download a file
url = "https://raw.githubusercontent.com/Utartizan/Quantum-Theory-GPT/refs/heads/main/pg66944.txt"
filename = wget.download(url)

print(f"File downloaded as {filename}")
```

``` python
with open('pg66944.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
```

``` python
print("length of dataset in characters is: ", len(text))

"""
Outputs the amount of characters within the attached dataset. 
In this case, usually the longer the length of the dataset the
better (regarding accuracy) the generative material is.

This is due to the quantity in training and validation data that the
model can use.
"""
```

_length of dataset in characters is:  391617_

_'Outputs the amount of characters within the attached dataset. In this case, usually the longer the length of the dataset the\nbetter (regarding accuracy) the generative material is. This is due to the quantity in training and validation data that the model can use.'_

``` python
print(text[:1000])
```

_Lord Kelvin writing in 1893, in his preface to the English edition of
    Hertz’s Researches on Electric Waves, says “many workers and many
    thinkers have helped to build up the nineteenth century school of
    _plenum_, one ether for light, heat, electricity, magnetism; and the
    German and English volumes containing Hertz’s electrical papers, given
    to the world in the last decade of the century, will be a permanent
    monument of the splendid consummation now realised.”_

_Ten years later, in 1905, we find Einstein declaring that “the ether
    will be proved to be superfluous.” At first sight the revolution in
    scientific thought brought about in the course of a single decade
    appears to be almost too violent. A more careful even though a rapid
    review of the subject will, however, show how the Theory of Relativity
    gradually became a historical necessity._

_Towards the beginning of the nineteenth century, the luminiferous ether
    came into prominence as a result of the brilliant successes of the wave
    theory_

``` python
"""
Scan the entire file (text) and output:
1. All the types of characters utilised
2. The quantifiable size of the characters utilised (94)
"""

characters = sorted(list(set(text)))
vocabularySize = len(characters)
print(''.join(characters))
print(vocabularySize)
```

#### This Outputs
     $&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}§°±²³´·¹½Ä×äæéôöüĀĊūΓΔΘΣΤΦΨΩαβγδεζηθκλμνξπρστφχψωḂḞḟṠṡṽẇῶ—‘’“”′″‴⁰⁴⁵⁷⁸⁻₀₁₂₃₄∂∑√∞∫∴≠□▽﻿
    176


``` python
"""
Implementing encoding and decoding functions for a character-level tokeniser

This essentially assigns each character to its assigned number, for every character
in the list is its own number.

In this case, the value 50 belongs to the character Q
the value 85 belongs to the character U
the value 1 belongs to the space character(?)

The print functions below exercises the model's ability to both encode and decode
a set of charaters, in the form of many words, in the form of a singular sentence 
accordingly.
"""

stoi = { ch:i for i,ch in enumerate(characters) }
itos = { i:ch for i,ch in enumerate(characters) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("A more careful even though a rapid review of the subject will, however, show how the Theory of Relativity gradually became a historical necessity."))
print(decode(encode("A more careful even though a rapid review of the subject will, however, show how the Theory of Relativity gradually became a historical necessity.")))
```

#### The 'print(encode...' line outputs..
    [30, 2, 73, 75, 78, 65, 2, 63, 61, 78, 65, 66, 81, 72, 2, 65, 82, 65, 74, 2, 80, 68, 75, 81, 67, 68, 2, 61, 2, 78, 61, 76, 69, 64, 2, 78, 65, 82, 69, 65, 83, 2, 75, 66, 2, 80, 68, 65, 2, 79, 81, 62, 70, 65, 63, 80, 2, 83, 69, 72, 72, 10, 2, 68, 75, 83, 65, 82, 65, 78, 10, 2, 79, 68, 75, 83, 2, 68, 75, 83, 2, 80, 68, 65, 2, 49, 68, 65, 75, 78, 85, 2, 75, 66, 2, 47, 65, 72, 61, 80, 69, 82, 69, 80, 85, 2, 67, 78, 61, 64, 81, 61, 72, 72, 85, 2, 62, 65, 63, 61, 73, 65, 2, 61, 2, 68, 69, 79, 80, 75, 78, 69, 63, 61, 72, 2, 74, 65, 63, 65, 79, 79, 69, 80, 85, 12]

#### The 'print(decode(encode(...' line outputs..    
    A more careful even though a rapid review of the subject will, however, show how the Theory of Relativity gradually became a historical necessity.


``` python
pip install torch
```


``` python
# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this
```

#### This outputs (and bear with me here...)

    torch.Size([391617]) torch.int64
    tensor([175,  41,  75,  78,  64,   2,  40,  65,  72,  82,  69,  74,   2,  83,
             78,  69,  80,  69,  74,  67,   2,  69,  74,   2,  15,  22,  23,  17,
             10,   2,  69,  74,   2,  68,  69,  79,   2,  76,  78,  65,  66,  61,
             63,  65,   2,  80,  75,   2,  80,  68,  65,   2,  34,  74,  67,  72,
             69,  79,  68,   2,  65,  64,  69,  80,  69,  75,  74,   2,  75,  66,
              1,  37,  65,  78,  80,  86, 149,  79,   2,  47,  65,  79,  65,  61,
             78,  63,  68,  65,  79,   2,  75,  74,   2,  34,  72,  65,  63,  80,
             78,  69,  63,   2,  52,  61,  82,  65,  79,  10,   2,  79,  61,  85,
             79,   2, 150,  73,  61,  74,  85,   2,  83,  75,  78,  71,  65,  78,
             79,   2,  61,  74,  64,   2,  73,  61,  74,  85,   1,  80,  68,  69,
             74,  71,  65,  78,  79,   2,  68,  61,  82,  65,   2,  68,  65,  72,
             76,  65,  64,   2,  80,  75,   2,  62,  81,  69,  72,  64,   2,  81,
             76,   2,  80,  68,  65,   2,  74,  69,  74,  65,  80,  65,  65,  74,
             80,  68,   2,  63,  65,  74,  80,  81,  78,  85,   2,  79,  63,  68,
             75,  75,  72,   2,  75,  66,   1,  60,  76,  72,  65,  74,  81,  73,
             60,  10,   2,  75,  74,  65,   2,  65,  80,  68,  65,  78,   2,  66,
             75,  78,   2,  72,  69,  67,  68,  80,  10,   2,  68,  65,  61,  80,
             10,   2,  65,  72,  65,  63,  80,  78,  69,  63,  69,  80,  85,  10,
              2,  73,  61,  67,  74,  65,  80,  69,  79,  73,  25,   2,  61,  74,
             64,   2,  80,  68,  65,   1,  36,  65,  78,  73,  61,  74,   2,  61,
             74,  64,   2,  34,  74,  67,  72,  69,  79,  68,   2,  82,  75,  72,
             81,  73,  65,  79,   2,  63,  75,  74,  80,  61,  69,  74,  69,  74,
             67,   2,  37,  65,  78,  80,  86, 149,  79,   2,  65,  72,  65,  63,
             80,  78,  69,  63,  61,  72,   2,  76,  61,  76,  65,  78,  79,  10,
              2,  67,  69,  82,  65,  74,   1,  80,  75,   2,  80,  68,  65,   2,
             83,  75,  78,  72,  64,   2,  69,  74,   2,  80,  68,  65,   2,  72,
             61,  79,  80,   2,  64,  65,  63,  61,  64,  65,   2,  75,  66,   2,
             80,  68,  65,   2,  63,  65,  74,  80,  81,  78,  85,  10,   2,  83,
             69,  72,  72,   2,  62,  65,   2,  61,   2,  76,  65,  78,  73,  61,
             74,  65,  74,  80,   1,  73,  75,  74,  81,  73,  65,  74,  80,   2,
             75,  66,   2,  80,  68,  65,   2,  79,  76,  72,  65,  74,  64,  69,
             64,   2,  63,  75,  74,  79,  81,  73,  73,  61,  80,  69,  75,  74,
              2,  74,  75,  83,   2,  78,  65,  61,  72,  69,  79,  65,  64,  12,
            151,   1,   1,  49,  65,  74,   2,  85,  65,  61,  78,  79,   2,  72,
             61,  80,  65,  78,  10,   2,  69,  74,   2,  15,  23,  14,  19,  10,
              2,  83,  65,   2,  66,  69,  74,  64,   2,  34,  69,  74,  79,  80,
             65,  69,  74,   2,  64,  65,  63,  72,  61,  78,  69,  74,  67,   2,
             80,  68,  61,  80,   2, 150,  80,  68,  65,   2,  65,  80,  68,  65,
             78,   1,  83,  69,  72,  72,   2,  62,  65,   2,  76,  78,  75,  82,
             65,  64,   2,  80,  75,   2,  62,  65,   2,  79,  81,  76,  65,  78,
             66,  72,  81,  75,  81,  79,  12, 151,   2,  30,  80,   2,  66,  69,
             78,  79,  80,   2,  79,  69,  67,  68,  80,   2,  80,  68,  65,   2,
             78,  65,  82,  75,  72,  81,  80,  69,  75,  74,   2,  69,  74,   1,
             79,  63,  69,  65,  74,  80,  69,  66,  69,  63,   2,  80,  68,  75,
             81,  67,  68,  80,   2,  62,  78,  75,  81,  67,  68,  80,   2,  61,
             62,  75,  81,  80,   2,  69,  74,   2,  80,  68,  65,   2,  63,  75,
             81,  78,  79,  65,   2,  75,  66,   2,  61,   2,  79,  69,  74,  67,
             72,  65,   2,  64,  65,  63,  61,  64,  65,   1,  61,  76,  76,  65,
             61,  78,  79,   2,  80,  75,   2,  62,  65,   2,  61,  72,  73,  75,
             79,  80,   2,  80,  75,  75,   2,  82,  69,  75,  72,  65,  74,  80,
             12,   2,  30,   2,  73,  75,  78,  65,   2,  63,  61,  78,  65,  66,
             81,  72,   2,  65,  82,  65,  74,   2,  80,  68,  75,  81,  67,  68,
              2,  61,   2,  78,  61,  76,  69,  64,   1,  78,  65,  82,  69,  65,
             83,   2,  75,  66,   2,  80,  68,  65,   2,  79,  81,  62,  70,  65,
             63,  80,   2,  83,  69,  72,  72,  10,   2,  68,  75,  83,  65,  82,
             65,  78,  10,   2,  79,  68,  75,  83,   2,  68,  75,  83,   2,  80,
             68,  65,   2,  49,  68,  65,  75,  78,  85,   2,  75,  66,   2,  47,
             65,  72,  61,  80,  69,  82,  69,  80,  85,   1,  67,  78,  61,  64,
             81,  61,  72,  72,  85,   2,  62,  65,  63,  61,  73,  65,   2,  61,
              2,  68,  69,  79,  80,  75,  78,  69,  63,  61,  72,   2,  74,  65,
             63,  65,  79,  79,  69,  80,  85,  12,   1,   1,  49,  75,  83,  61,
             78,  64,  79,   2,  80,  68,  65,   2,  62,  65,  67,  69,  74,  74,
             69,  74,  67,   2,  75,  66,   2,  80,  68,  65,   2,  74,  69,  74,
             65,  80,  65,  65,  74,  80,  68,   2,  63,  65,  74,  80,  81,  78,
             85,  10,   2,  80,  68,  65,   2,  72,  81,  73,  69,  74,  69,  66,
             65,  78,  75,  81,  79,   2,  65,  80,  68,  65,  78,   1,  63,  61,
             73,  65,   2,  69,  74,  80,  75,   2,  76,  78,  75,  73,  69,  74,
             65,  74,  63,  65,   2,  61,  79,   2,  61,   2,  78,  65,  79,  81,
             72,  80,   2,  75,  66,   2,  80,  68,  65,   2,  62,  78,  69,  72,
             72,  69,  61,  74,  80,   2,  79,  81,  63,  63,  65,  79,  79,  65,
             79,   2,  75,  66,   2,  80,  68,  65,   2,  83,  61,  82,  65,   1,
             80,  68,  65,  75,  78,  85])
:::
:::

``` python
# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
trainingData = data[:n]
validationData = data[n:]
```

``` python
block_size = 8
trainingData[:block_size+1]
```

#### This outputs:

    tensor([175,  41,  75,  78,  64,   2,  40,  65,  72])

:::

``` python
x = trainingData[:block_size]
y = trainingData[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
```

::: {.output .stream .stdout}
    when input is tensor([175]) the target: 41
    when input is tensor([175,  41]) the target: 75
    when input is tensor([175,  41,  75]) the target: 78
    when input is tensor([175,  41,  75,  78]) the target: 64
    when input is tensor([175,  41,  75,  78,  64]) the target: 2
    when input is tensor([175,  41,  75,  78,  64,   2]) the target: 40
    when input is tensor([175,  41,  75,  78,  64,   2,  40]) the target: 65
    when input is tensor([175,  41,  75,  78,  64,   2,  40,  65]) the target: 72
:::

``` python
"""
This batch of code illustrates how sequences are batched 
and how each prediction is made based on context.

For each sequence in the batch, we're looking at how the model would predict
the next character/token on what it has gone through so far.

This would allow for the model to train (learn the dependencies in sequences).
"""


torch.manual_seed(6969) # setting a completely random (lol funny i know) number to allow for reproducibility
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions? how many elements used to predict the next?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y for the model
    data = trainingData if split == 'train' else validationData
    #It chooses between trainingData or validationData based on the split parameter.
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    """
    Generates random starting indices for data samples.
    The subtraction of block_size ensures that there's enough data for each sequence.
    """

    x = torch.stack([data[i:i+block_size] for i in ix])
    
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

xb, yb = get_batch('train')

print('-------')

print('inputs:')
print(xb.shape)
print(xb)

print('-------')

print('targets:')
print(yb.shape)
print(yb)

print('----------------')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
```

    -------
    inputs:
    torch.Size([4, 8])
    tensor([[69, 78, 79, 80,  2, 80, 68, 65],
            [ 2, 78, 65, 72, 61, 80, 69, 82],
            [ 2, 80, 68, 65,  2, 65, 77, 81],
            [65, 78, 61, 62, 72, 65,  2, 73]])
    -------
    targets:
    torch.Size([4, 8])
    tensor([[78, 79, 80,  2, 80, 68, 65, 78],
            [78, 65, 72, 61, 80, 69, 82, 69],
            [80, 68, 65,  2, 65, 77, 81, 61],
            [78, 61, 62, 72, 65,  2, 73, 61]])
    ----------------
    when input is [69] the target: 78
    when input is [69, 78] the target: 79
    when input is [69, 78, 79] the target: 80
    when input is [69, 78, 79, 80] the target: 2
    when input is [69, 78, 79, 80, 2] the target: 80
    when input is [69, 78, 79, 80, 2, 80] the target: 68
    when input is [69, 78, 79, 80, 2, 80, 68] the target: 65
    when input is [69, 78, 79, 80, 2, 80, 68, 65] the target: 78
    when input is [2] the target: 78
    when input is [2, 78] the target: 65
    when input is [2, 78, 65] the target: 72
    when input is [2, 78, 65, 72] the target: 61
    when input is [2, 78, 65, 72, 61] the target: 80
    when input is [2, 78, 65, 72, 61, 80] the target: 69
    when input is [2, 78, 65, 72, 61, 80, 69] the target: 82
    when input is [2, 78, 65, 72, 61, 80, 69, 82] the target: 69
    when input is [2] the target: 80
    when input is [2, 80] the target: 68
    when input is [2, 80, 68] the target: 65
    when input is [2, 80, 68, 65] the target: 2
    when input is [2, 80, 68, 65, 2] the target: 65
    when input is [2, 80, 68, 65, 2, 65] the target: 77
    when input is [2, 80, 68, 65, 2, 65, 77] the target: 81
    when input is [2, 80, 68, 65, 2, 65, 77, 81] the target: 61
    when input is [65] the target: 78
    when input is [65, 78] the target: 61
    when input is [65, 78, 61] the target: 62
    when input is [65, 78, 61, 62] the target: 72
    when input is [65, 78, 61, 62, 72] the target: 65
    when input is [65, 78, 61, 62, 72, 65] the target: 2
    when input is [65, 78, 61, 62, 72, 65, 2] the target: 73
    when input is [65, 78, 61, 62, 72, 65, 2, 73] the target: 61
    
``` python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(6969)


"""
Creating the definition for the Model (BLModel), essentially creating an embedding
layer where each token is assigned/mapped to a vector of the same identity as the
vocabulary.

This is to allow each token to represent a learnable table where each entry 
will correspond to X logits for the corresponding token.
"""

class BLModel(nn.Module):
    def __init__(self, vocabularySize):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabularySize, vocabularySize)

    """
    
    """
    
    def forward(self, idx, targets=None):


        """
        Logits are unnormalised predictions that would be output by this model
        for each class in a classification problem before they're transformed
        into probabiltiies.

        They operate on an inherently unlimited scale, being any range of 
        values whether positive or negative.

        In this example these logits will be converted into a score between 0 
        and 1 to represent the validity or probability distribution over each
        possible output.
        """

        """
        
        -- Refer to 
        -- https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        
        """
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


"""
From the following results you'll notice that the generated text will form to become
a string of characters from the BLModel (so  far).

However due to there being no clear semantic nor syntactic structure, we can
classify that:
1. Not well-trained for text generation
2. Dataset includes a lot of noise or non-standard text formatting
3. Process does not implement any filtering out of non-meaningful characters.

The loss value of 5.5570 is slightly over the expected loss value from the following
calculation of ln(176) which is equal to 5.17048399504.

What's important to know is that losses can vary from batch to batch, or the
issues in data (noise, which in this case would be any extremely hard-to-understand symbols
from the perspective of the model (e.g. calculus formulae))

We are however getting somewhere.
"""


m = BLModel(vocabularySize)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

### The output of which is:
    torch.Size([32, 176])
    tensor(5.5570, grad_fn=<NllLossBackward0>)
    λ&b3?⁻ô∞xḂxge√&√ξΣ²φn▽g′θBβt₄s³Eα-JKsψV₀ḂR²“;,psΨU5CJö(7ḟ∑Tη9üfΤẇHuD,{▽4du&4γcĀq.RQρ⁴N$8p·“φ⁰ä5′Yyε9
