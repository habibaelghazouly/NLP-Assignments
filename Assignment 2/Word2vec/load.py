import numpy as np
import pickle

def load_embeddings_and_vocab(emb_path="embeddings.npy",
                              word2idx_path="word2idx.pkl",
                              idx2word_path="idx2word.pkl",
                              device="cpu"):
    # Load embeddings
    embeddings = np.load(emb_path)

    # Load vocabularies
    with open(word2idx_path, "rb") as f:
        word2idx = pickle.load(f)

    with open(idx2word_path, "rb") as f:
        idx2word = pickle.load(f)

    return embeddings, word2idx, idx2word