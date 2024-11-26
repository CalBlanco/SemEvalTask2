
from collaters import Collator
from datasets import TagingDataset
import torch
from torch import nn
from torch.nn import GRU
#from data_util import retrieve_coNER
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.metrics import f1_score
import json
import os


#model definition
class GRUModel(nn.Module):
    def __init__(self, token_vocab, tag_vocab, embedding_dim=512, hidden_dim=1024):
        """ GRU Model for token tagging

        **ARGS**
            token_vocab: dictionary of tokens with their token id as a number [REQUIRED]
            tag_vocab: dictionary of tags with their tag id as a number [REQUIRED]
            embedding_dim: Int size for embedding dimension [DEFAULT=512]
            hidden_dim: Int size for hidden dimenesion [DEFAULT=1024]

        *Notes*:
            Will automatically pick the following devices in this order if the devices are available
            1. Cuda (gpu)
            2. mps (apple sillicon/metal)
            3. cpu
        
        **RETURNS**
            An instance of the GRU Model with desired parameters
        """
        super(GRUModel, self).__init__()
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.device = torch.device('cuda') if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else torch.device('cpu') #try GPU, then MPS, then CPU if none are available

        self.decode_tokens = {int(v):k for k,v in token_vocab.items()}
        self.decode_tags = {int(v):k for k,v in tag_vocab.items()}


        self.embedding = nn.Embedding(len(token_vocab), embedding_dim, padding_idx=0)
        self.gru = GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(tag_vocab))

    def forward(self, x):
        embeddings = self.embedding(x)
        out, _ = self.gru(embeddings)
        out = self.fc(out)
        return out
    

    def fit(self, train:DataLoader, val:DataLoader, epochs=30, learning_rate=0.0001):
        """ Main training loop for the GRU Model

        **ARGS**
            train: DataLoader to pull training samples from [REQUIRED]
            val: DataLoader to pull validation samples from [REQUIRED]
            epochs: integer count for epoch number [DEFUALT=30]
            learning_rate: float for learning rate steps [DEFAULT=0.0001]

        """
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tag_vocab['<PAD>'])
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        model = self.to(self.device)

        metrics = []
        best_f1 = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        # Training Loop
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            for token_ids, tag_ids in train:
                token_ids = token_ids.to(self.device)
                tag_ids = tag_ids.to(self.device)

                optimizer.zero_grad()

                outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # Validation
            model.eval()
            total_val_loss = 0
            all_predictions = []
            all_tags = []

            with torch.no_grad():
                for token_ids, tag_ids in val:
                    token_ids = token_ids.to(self.device)
                    tag_ids = tag_ids.to(self.device)

                    outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

                    outputs = outputs.view(-1, outputs.shape[-1])
                    tag_ids = tag_ids.view(-1)
                    loss = loss_fn(outputs, tag_ids)
                    total_val_loss += loss.item()

                    predictions = outputs.argmax(dim=1)
                    mask = tag_ids != self.tag_vocab['<PAD>']

                    all_predictions.extend(predictions[mask].tolist())
                    all_tags.extend(tag_ids[mask].tolist())

            # compute train and val loss
            train_loss = total_train_loss / len(train)
            val_loss = total_val_loss / len(val)

            # Calculate F1 score
            f1 = f1_score(all_tags, all_predictions, average='macro')
            metrics.append([epoch+1, train_loss, val_loss, f1])
            if f1 > best_f1:
                best_f1 = f1
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if train_loss < best_train_loss:
                        torch.save(model.state_dict(), "best_model.pt")
                        best_train_loss = train_loss
            print(f'epoch = {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

    def predict(self, test:DataLoader)->list[tuple[list[int], list[int]]]:
        """ Given a DataLoader to get inputs from output a list of predictions

            **RETURNS**
                A list of tuples of the form [ ([token_0, token_1, ... token_n], [tag_0, tag_1, ... tag_n]), ... ]
        """
        all_predictions = []
        self.eval()
        with torch.no_grad():
            for token_ids, _ in test:
                token_ids = token_ids.to(self.device)
                outputs = self(token_ids) # (batch_size, seq_len, tagset_size)
                predictions = outputs.argmax(dim=-1)  # Get predictions for each token
                for i, predict in enumerate(predictions):
                    all_predictions.append((token_ids[i].tolist(), predict.tolist()))
               # all_predictions.append(predictions.view(-1).tolist())
        return all_predictions
    
    def decode(self, predicitions:list[tuple[list[int], list[int]]])->list[tuple[list[str], list[str]]]:
        """ Given an input predicition of the shape [ ([token_0, token_1, ... token_n], [tag_0, tag_1, ... tag_n]), ... ] decode the 
            tokens and the tags based on our models decoding tables.

            *notes*:
                Will remove padding tokens and output the sequences without pads.

            **RETURNS**:
                a list of decoded predictions in the form [ ([decoded_token_0, decoded_token_1, ... decoded_token_n], [decoded_tag_0, decoded_tag_1, ... decoded_tag_n])]

        """
        decoded = []
        for tokens, tags in predicitions:
            decoded_tokens = [self.decode_tokens[x] for x in tokens if x!=self.token_vocab['<PAD>']]
            decoded_tags = [self.decode_tags[x] for x in tags if x!=self.tag_vocab['<PAD>']]

            decoded.append((decoded_tokens, decoded_tags))

        return decoded

