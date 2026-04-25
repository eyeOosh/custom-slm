# # # # # # # # # # # # # # # # # # # # # # # # # # #
# Aayush Kumar                                      #
# All comments are notes for me                     #
# Learning how to make my own Small Language Model  #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import re

# 1. load in text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(len(raw_text))
print(raw_text[:99])

# 2. Tokenization time
preproc = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preproc = [item.strip() for item in preproc if item.strip()]
print(preproc[:30])
print("Length of text:", len(preproc))

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
