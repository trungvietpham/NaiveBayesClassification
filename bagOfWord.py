from turtle import pd


import pandas as pd
import ast
import numpy as np

folder = "data/"

data = pd.read_csv(folder + "email.csv")
word = open(folder+"vocabulary.txt", "r")
content = word.read()
vocabulary = ast.literal_eval(content)

X = np.zeros((data.shape[0], len(vocabulary))) #save frequence of each word
y = np.zeros((data.shape[0])) #save class

for i in range(data.shape[0]):
  email = data.iloc[i,0].split()
  for email_word in email:
    if email_word.lower() in vocabulary:
      X[i, vocabulary[email_word]] += 1
      y[i] = data.iloc[i,1]
      

np.save(folder+"X.npy", X)
np.save(folder+"y.npy", y)
