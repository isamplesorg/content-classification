import numpy as np
import pandas as pd


rand_state = int(43)
samplesize = int(1200)
classname = ""
traintextcol = "trainingtext"
datadir = "C:/Users/smrTu/OneDrive/Documents/Workspace/iSamples/training/"
the_df = pd.DataFrame()
next_df = pd.DataFrame()

df = pd.read_csv(datadir+'trainingdata-part1.csv', usecols=['igsn', traintextcol],dtype={'igsn':str,  traintextcol:str})
next_df = df.sample(n=samplesize, random_state=22)
the_df = pd.concat([the_df, next_df])

df = pd.read_csv(datadir+'trainingdata-part2.csv', usecols=['igsn', traintextcol],dtype={'igsn':str,  traintextcol:str})
next_df = df.sample(n=samplesize, random_state=23)
the_df = pd.concat([the_df, next_df])

df = pd.read_csv(datadir+'trainingdata-part3.csv', usecols=['igsn', traintextcol],dtype={'igsn':str,  traintextcol:str})
next_df = df.sample(n=samplesize, random_state=24)
the_df = pd.concat([the_df, next_df])

df = pd.read_csv(datadir+'trainingdata-part4.csv', usecols=['igsn', traintextcol],dtype={'igsn':str,  traintextcol:str})
next_df = df.sample(n=samplesize, random_state=25)
the_df = pd.concat([the_df, next_df])

df = pd.read_csv(datadir+'trainingdata-part5.csv', usecols=['igsn', traintextcol],dtype={'igsn':str,  traintextcol:str})
next_df = df.sample(n=samplesize, random_state=26)
the_df = pd.concat([the_df, next_df])

the_df.to_csv(datadir+'train_df.csv')

print("all done")

