import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.downloader import load

#model definition
class TransformerNER(nn.Module):
    def __init__(self, token_vocab, tag_vocab, embedding_dim=512, hidden_dim=1024, dropout=0.3):
        """ Transformer Model for token tagging

        **ARGS**
            token_vocab: dictionary of tokens with their token id as a number [REQUIRED]
            tag_vocab: dictionary of tags with their tag id as a number [REQUIRED]
            embedding_dim: Int size for embedding dimension [DEFAULT=512]
            hidden_dim: Int size for hidden dimenesion [DEFAULT=1024]
            dropout: Float for dropout rate [DEFAULT=0.3]
        *Notes*:
            Will automatically pick the following devices in this order if the devices are available
            1. Cuda (gpu)
            2. mps (apple sillicon/metal)
            3. cpu
        
        **RETURNS**
            An instance of the GRU Model with desired parameters
        """
        super(TransformerNER, self).__init__()
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.device = torch.device('cuda') if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else torch.device('cpu') #try GPU, then MPS, then CPU if none are available

        self.decode_tokens = {int(v):k for k,v in token_vocab.items()}
        self.decode_tags = {int(v):k for k,v in tag_vocab.items()}


        self.embedding = nn.Embedding(len(token_vocab), embedding_dim, padding_idx=0)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim, len(tag_vocab))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None):
        """
        x: input token ids
        y: target token ids (only needed during training)
        """
        # During inference, we'll use the input as both src and tgt
        if y is None:
            y = x

        # Create padding masks
        src_key_padding_mask = (x == self.token_vocab['<PAD>'])  # (batch_size, src_len)
        tgt_key_padding_mask = (y == self.tag_vocab['<PAD>'])    # (batch_size, tgt_len)
        
        # Create causal mask for decoder to prevent looking at future tokens
        tgt_mask = self.transformer.generate_square_subsequent_mask(y.size(1)).to(x.device)
        
        # Get embeddings
        src_embeddings = self.embedding(x)
        tgt_embeddings = self.embedding(y)
        
        # Pass through transformer
        out = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            tgt_mask=tgt_mask,                    # Causal mask for decoder
            src_key_padding_mask=src_key_padding_mask,  # Padding mask for encoder
            tgt_key_padding_mask=tgt_key_padding_mask   # Padding mask for decoder
        )
        
        out = self.dropout(out)
        out = self.fc(out)
        return out
    

    def fit(self, train:DataLoader, val:DataLoader, epochs=30, learning_rate=0.0001):
        """ Main training loop for the Transformer Model

        **ARGS**
            train: DataLoader to pull training samples from [REQUIRED]
            val: DataLoader to pull validation samples from [REQUIRED]
            epochs: integer count for epoch number [DEFUALT=30]
            learning_rate: float for learning rate steps [DEFAULT=0.0001]

        """
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tag_vocab['<PAD>'])
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        model = self.to(self.device)

        metrics = []
        # Training Loop
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            for token_ids, output_ids in train:
                token_ids = token_ids.to(self.device)
                output_ids = output_ids.to(self.device)

                optimizer.zero_grad()

                predictions = model(token_ids, token_ids)  # (batch_size, seq_len, vocab_size)

                # Reshape for loss computation
                predictions = predictions.reshape(-1, predictions.shape[-1])
                targets = output_ids.reshape(-1)

                # Compute loss
                loss = loss_fn(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            total_val_loss = 0
            all_predictions = []
            all_output_ids = []

            with torch.no_grad():
                for token_ids, output_ids in val:
                    token_ids = token_ids.to(self.device)
                    output_ids = output_ids.to(self.device)

                    outputs = model(token_ids)

                    predictions = outputs

                    # Reshape for loss computation
                    predictions = predictions.reshape(-1, predictions.shape[-1])
                    targets = output_ids.reshape(-1)

                    # Compute loss
                    loss = loss_fn(predictions, targets)
                    total_val_loss += loss.item()

                    predictions = outputs.argmax(dim=-1)
                    mask = output_ids != self.tag_vocab['<PAD>']

                    all_predictions.extend(predictions[mask].tolist())
                    all_output_ids.extend(output_ids[mask].tolist())

            # compute train and val loss
            train_loss = total_train_loss / len(train)
            val_loss = total_val_loss / len(val)

            # Calculate metrics
            metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
            })
            print(f'epoch = {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f}')
            torch.save(model.state_dict(), f"transformer_ner_{epoch+1}.pt")
        # After training loop
        train_losses = [m['train_loss'] for m in metrics]
        val_losses = [m['val_loss'] for m in metrics]
        epochs = range(1, epochs + 1)

        plt.figure(figsize=(20, 10))
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()

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
    
    def decode(self, predicitions:list[tuple[list[int], list[int]]], target_sentences:list[list[str]])->list[tuple[list[str], list[str]]]:
        """ Given an input predicition of the shape [ ([token_0, token_1, ... token_n], [tag_0, tag_1, ... tag_n]), ... ] decode the 
            tokens and the tags based on our models decoding tables.

            *notes*:
                Will remove padding tokens and output the sequences without pads.

            **RETURNS**:
                a list of decoded predictions in the form [ ([decoded_token_0, decoded_token_1, ... decoded_token_n], [decoded_tag_0, decoded_tag_1, ... decoded_tag_n])]

        """
        decoded = []
        for i, (tokens, tags) in enumerate(predicitions):
            decoded_tokens = []
            for j, x in enumerate(tokens):
                if x != self.token_vocab['<PAD>']:
                    if x != self.token_vocab['<UNK>']:
                        decoded_tokens.append(self.decode_tokens[x])
                    else:
                        decoded_tokens.append(target_sentences[i][j])
                
            decoded_tags = [self.decode_tags[x] for x in tags if x!=self.tag_vocab['<PAD>']]

            decoded.append((decoded_tokens, decoded_tags))

        return decoded
    
    def load_model(self, path:str):
        """ Load a pre-trained model from a given path

        **ARGS**
            path: string path to the pre-trained model
        """
        self.to(self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)