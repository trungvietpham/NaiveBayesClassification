import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words

vocabulary = {}
data = pd.read_csv("data/email.csv")
nltk.download('words')
set_words = set(words.words())

def build_vocabulary(current_email):
  idx = len(vocabulary)
  for word in current_email:
    if word.lower() not in vocabulary and word.lower() in set_words:
      vocabulary[word] = idx
      idx+=1

if __name__ == '__main__':
  for i in range(data.shape[0]):
    current_email = data.iloc[i,0].split()
    print(f'Current email: {i}/{data.shape[0]}, length of vocabulary: {len(vocabulary)}')
    build_vocabulary(current_email)

  file = open("data/vocabulary.txt", "w")
  file.write(str(vocabulary))
  file.close()