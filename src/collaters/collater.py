from torch.nn.utils.rnn import pad_sequence


class Collator():
   
    def __init__(self, token_vocab, tag_vocab):
        """ Collator wrapper to pass needed data before generating collate function
    
        **ARGS**
            token_vocab: dictionary of token numerical ids [REQUIRED]
            tag_vocab: dictionary of tag numerical ids [required]
        """
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab


    # collate token_ids and tag_ids to make mini-batches
    def collate_fn(self, batch):
        """
            Pass this bad boi to the DataLoader as the collate_fn
        """
        # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]

        # Separate sentences and tags
        token_ids = [item[0] for item in batch]
        tag_ids = [item[1] for item in batch]

        # Pad sequences
        sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=self.token_vocab['<PAD>'])
        # sentences_pad.size()  (batch_size, seq_len)
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=self.tag_vocab['<PAD>'])
        # tags_pad.size()  (batch_size, seq_len)
        return sentences_padded, tags_padded
