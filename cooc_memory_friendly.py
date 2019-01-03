#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle


### ------ a version that works with less memory ------------###
def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col= [], [], []
    # use compressed sparce matrix instead of coo_matrix
    cooc = coo_matrix((vocab_size, vocab_size))
    counter = 1
    for fn in ['data/train_pos_full.txt', 'data/train_neg_full.txt']:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
    
                # add entries to compressed matrix to save memory
                if counter % 250000 == 0:
                    cooc += coo_matrix((data, (row, col)), shape=(vocab_size, vocab_size))
                    data, row, col = [], [], []

                counter += 1

            cooc += coo_matrix((data, (row, col)), shape=(vocab_size, vocab_size))
            data, row, col = [], [], []

    # somehow adding coo matrices converts them to csr type s pare matrices.
    # convert back to coo format
    cooc = cooc.tocoo()               
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
