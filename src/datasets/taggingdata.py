import torch
from torch.utils.data import Dataset

class TagingDataset(Dataset):
    def __init__(self, x, y, token_vocab=None, tag_vocab=None, training=True):
        """ Generic Tagging dataset

        **ARGS**
            x: iterable containing input for dataset [REQUIRED]
            y: iterable contaiing output for dataset [REQUIRED]
            token_vocab: dictionary of tokens and their numerical ids [DEFAULT=None]
            tag_vocab: dictionary of tags and their numerical ids [DEFAULT=None]
            training: boolean to determine if we are training [DEFAULT=True]

        *Notes*
            If training is true (default setting) it will build a token, and tag vocab to be used in the validation and testing sets

        """
        # Create vocabularies if training
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0, '<UNK>': 1}

            # build vocab from training data
            for i in range(len(x)):
                for token in x[i]:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                for tag in y[i]:
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        # Convert sentences and tags to integer IDs during initialization
        self.corpus_token_ids = []
        self.corpus_tag_ids = []
        for i in range(len(x)):
            token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in x[i]]
            tag_ids = [self.tag_vocab.get(tag, self.tag_vocab['<UNK>']) for tag in y[i]]
            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_tag_ids.append(torch.tensor(tag_ids))

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]
