# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Aayush Kumar                                      #
# All comments are notes for me                     #
# Learning how to make my own Small Language Model  #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import re

import torch

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
import torch

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, maxLen, stride):
        self.inputIds = []
        self.targetIds = []

        tokenIds = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # input ids are the first 4 tokens, target ids are the next 4 tokens
        # then we shift by stride and repeat
        for i in range(0, len(tokenIds) - maxLen, stride):
            inputId = tokenIds[i:i+maxLen]
            targetId = tokenIds[i+1:i+maxLen+1]

            self.inputIds.append(inputId)
            self.targetIds.append(targetId)

    def __len__(self):
        return len(self.inputIds)
    
    def __getitem__(self, idx):
        # By returning tensors here, the DataLoader knows exactly how to stack them into a batch
        return torch.tensor(self.inputIds[idx]), torch.tensor(self.targetIds[idx])

# dataSet is GPT dataset, dataloader is a wrapper around the dataset that allows us to iterate through it in batches
# batch size is how many batches the model processes at once, before updating the weights
# maxLen is the context size, stride is how much we shift the window by to create the next input-target pair
# numWorkers is how many parallel processes to use for loading the data, 0 means use the main process
def createDataLoader(txt, batchSize=4, maxLen=256, stride=128, shuffle=True, dropLast=True, numWorkers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, maxLen, stride)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, drop_last=dropLast, num_workers=numWorkers)
    return dataloader

# lets test dataloader with batch size 1 and context size 4
print("\n" + f"="*50)
print("Part 5: DataLoader")
print(f"="*50, "")

# input size/context size of 4 is pretty small, 256 is more standard
dataloader = createDataLoader(raw_text, batchSize=1, maxLen=4, stride=1, shuffle=False)
dataIter = iter(dataloader)
firstBatch = next(dataIter)
print("Batch 1 (size 1) (input ids, target ids):\n", firstBatch, "\n")
print("Batch 2 (size 1) (input ids, target ids):\n", next(dataIter), "\n")

# stride = 4, batch size = 8
# 8 arrays in tensor, and inputIds and targetIds are shifted by 4 tokens
# so no overlap in inputs, and no overlaps in inputs tensor and targets tensor
dataloader8 = createDataLoader(raw_text, batchSize=8, maxLen=4, stride=4, shuffle=False)
dataIter8 = iter(dataloader8)
inputIds8, targetIds8 = next(dataIter8)
print("Batch 1 (size 8, stride 4) input ids:\n", inputIds8, "\n")
print("Batch 1 (size 8, stride 4) target ids:\n", targetIds8, "\n")

print("="*50)
print("Part 6: Token / Vector Embeddings")
print("="*50, "")
# convert input-target pairs into vector embeddings
# these token embeddings are the input to the model
# currently token ids are just ramdom numbers
# for ex. cat and kitten are semantically realed but
# their token id cat = 34, kitten = -13 does NOT represent any relatoinship
# vector embeddings represent these relationships
# the closer two vectors are, the closer they are in vector space


