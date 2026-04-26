# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Aayush Kumar                                      #
# All comments are notes for me                     #
# Learning how to make my own Small Language Model  #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import re

print("\nLearning LLMs from Scratch - Aayush Kumar")

print(f"="*50)
print("Part 1: Load in Text")
print(f"="*50, "")
# 1. load in text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(len(raw_text))
print(raw_text[:99])

print("\n" + f"="*50)
print("Part 2: Tokenizing")
print(f"="*50, "")
# 2. Tokenization time
preproc = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preproc = [item.strip() for item in preproc if item.strip()]
print(preproc[:30])
print("Length of text:", len(preproc))

print("\n" + f"="*50)
print("Part 3: Encoding")
print(f"="*50, "")
# 3. Convert tokens into token IDs - Encoding
# vocabulary is a dictionary of unique tokens sorted alphabetically
# then we assign token IDs to the vocab
allWords = sorted(set(preproc))
vocabSize = len(allWords)

print("Length of vocabulary:", vocabSize)

vocabulary = {token:integer for integer,token in enumerate(allWords)}

# create a class to use tokenizers easier
class Tokenizer:
    def __init__(self, vocab):
        self.strToInt = vocab
        self.intToStr = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproc = [item.strip() for item in preproc if item.strip()]
        ids = [self.strToInt[s] for s in preproc]
        return ids

    def decode(self, ids):
        text = " ".join([self.intToStr[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = Tokenizer(vocabulary)
test = "It's the last time he painted, you know"
ids = tokenizer.encode(test)
print("IDs:", ids)
print(tokenizer.decode(ids))

# this works on words in the vocab, but not on ones that are not
# add special context tokens to deal with words not in vocab
# <|unk|> and <|endoftext|> for unknown and end of text tokens
allWords.extend(["<|endoftext|>", "<|unk|>"])
vocabulary = {token:integer for integer,token in enumerate(allWords)}
print("Length of vocabulary:", len(allWords))

class Tokenizer2:
    def __init__(self, vocab):
        self.strToInt = vocab
        self.intToStr = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproc = [item.strip() for item in preproc if item.strip()]
        preproc = [item if item in self.strToInt
                   else "<|unk|>" for item in preproc]
        ids = [self.strToInt[s] for s in preproc]
        return ids

    def decode(self, ids):
        text = " ".join([self.intToStr[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer2 = Tokenizer2(vocabulary)
text1 = "Hello, I'm a human"
text2 = "In the midsummer solstice sulit terraces"

print("\n" + f"="*50)
print("Part 3.1: Encoding with <|unk|> and <|endoftext|>")
print(f"="*50, "")

text = " <|endoftext|> ".join((text1, text2))
ids2 = tokenizer2.encode(text)
print("IDs (2):", ids2)
print(tokenizer2.decode(ids2))

# note other special tokens
# BOS - Beginning of sequence
# EOS - End of sequence
# PAD - padding
# but I'm not gonna use those for what I'm tryna do

# now time to learn byte-pair encoding, which is how GPT tokenizes
# sub-word pased tokenization: do not split frequently used words into smaller words
#                              split rare words into smaller, meaningful subwords

# most common pair of consecutive bytes is replaced by a byte that does not occur in the data
# aaabdaaabac - aa is replaced by Z
# ZabdZabac - replace ab by Y
# ZYdZYac - done first layer, now ZY replaced W
# WdWac - no pairs to replace

# in set ("old</w>": 7, "older</w>": 3, "finest</w>": 9, "lowest</w>": 4)
# we can take char level tokenization, of which we find that the pairing 'es' appears the most
# so make 'es' a new token, then find that 'es' and 't' appear the most
# so make 'est' a new token, then find that 'est' and '</w>' appear the most
# so make 'est</w>' a new token, then find that 'o' and 'l' appear the most
# so make 'ol' a new token, then find that 'ol' and 'd' appear the most
# so make 'old' a new token, and remove all tokens with 0 apperances
# use tiktoken
import tiktoken
import importlib
tiktokenizer = tiktoken.get_encoding("gpt2")

print("\n" + f"="*50)
print("Part 3.2: Encoding with GPT-2 BPE using TikToken")
print(f"="*50, "")

text = "Hello matey! <|endoftext|> Waddup matey! What a great day, innit?"
idsTT = tiktokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(idsTT)
stringsTT = tiktokenizer.decode(idsTT)
print(stringsTT)

# note that this doesn't use <|unk|> token because its breaking down words to subwords and characters
# encoding some nonsense like 'Akwirw ier' is still broken down correctly
# like maybe 'ir' and 'er' are commonly occuring, which get replaced, and everything else as well
# 50,256 total tokens in gpt2 and gpt3 BPE are encompassing everything in the 170k-250k words in english


# 4. Input Target Pairs
encodedText = tiktokenizer.encode(raw_text)
print("Number of tokens in the raw text (aka Vocabulary size):", len(encodedText))
encSample = encodedText[50:] # remove first 50 tokens from the data set

contextSize = 4 #length of input
# model is trained to look at 4 tokens to predict the next, 5th token
print("\n" + f"="*50)
print(f"Part 4: Input Target Pairs with a context size {contextSize}")
print(f"="*50, "")

x = encSample[:contextSize]
y = encSample[1:contextSize + 1]

print(f"x: {x}")
print(f"y:      {y}")
#x: [290, 4950, 2241, 287]
#y:      [4950, 2241, 287, 257]
# 290 --> 4950
# 290, 4950 --> 2241
# 290, 4950, 2241 --> 287
# 290, 4950, 2241, 287 --> 257

for i in range(1, contextSize + 1):
    context = encSample[:i]
    desired = encSample[i]
    print("Input:", context, "  --> ", "Target:", desired)

# now use dataset and dataloaders to make these arrays into tensors for PyTorch
# sample text: 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
# x = tensor([[a, b, c, d],
#             [e, f, g, h],
#             [    ...   ]])
# y = tensor([[b, c, d, e],
#             [f, g, h, i],
#             [    ...   ]])

from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, maxLen, stride):
        self.inputIds = []
        self.targetIds = []
        