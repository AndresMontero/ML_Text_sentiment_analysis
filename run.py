import numpy as np
import pickle
import torch
import torch.utils.data as utils
from helpers import *

path_embeddings = 'embeddings/embeddings200_pretrained.npy'
path_vocab = 'embeddings/vocab_pretrained.pkl'
path_test = 'data/test_data.txt'
path_net = 'final_submission_v3.pt'
submission_filename = 'best_submission.csv'

#load the neural net
net = torch.load(path_net)

# load word embeddings
embeddings = np.load(path_embeddings)
# add line of zeroes to the embeddings for empty words
embeddings = np.append(np.zeros((1, embeddings.shape[1])), embeddings, axis=0)
# load vocabulary
with open(path_vocab, 'rb') as f:
    vocab = pickle.load(f)

# find longest tweet
longest = 0
with open(path_test) as f:
    for line in f:
        length = len(line.strip().split())
        if length > longest:
            longest = length          
print("Longest tweet has {:d} words".format(longest))

# Process test tweets
x = []
print("Loading test tweets..")
with open(path_test) as f:
    for line in f:
        tweet = np.int32(np.zeros((longest)))
        wordcount = 0
        # filter out the IDs and first comma
        line_bare = line[(line.index(",")+1):]
        for word in line_bare.strip().split():
            index = vocab.get(word, -1);
            # skip words for which we have no embedding
            if(index != -1):
                tweet[wordcount] = index + 1
                wordcount += 1
        x.append(tweet)     
x_test = np.asarray(x)

# Compute preditions
print("Running prediction..")
net.eval()
batch_size = 1024
# create a dataloader to iterate over the test data
test_loader = utils.DataLoader(torch.from_numpy(x_test), batch_size, shuffle = False)
submission_labels = np.zeros((0))
for tweets in iter(test_loader):
    predictions = net.predict(tweets.long())
    #conversion from (0, 1) to (-1, 1)
    labels = predictions.data.numpy() * 2 - 1
    submission_labels = np.concatenate((submission_labels, labels), axis=0)
    
# we need to add IDs to meet the submission interface requirements
ids = np.arange(len(submission_labels)) + 1
create_csv_submission(ids, submission_labels, submission_filename)
print("Submissions saved as " + submission_filename)

