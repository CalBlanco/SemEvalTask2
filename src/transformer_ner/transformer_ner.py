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

    def forward(self, x, y):
        embeddings = self.embedding(x)
        target_embeddings = self.embedding(y)
        out = self.transformer(embeddings, target_embeddings)
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
        #best_f1 = 0
        #best_val_loss = float('inf')
        #best_train_loss = float('inf')
        # Training Loop
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            for token_ids, output_ids in train:
                token_ids = token_ids.to(self.device)
                output_ids = output_ids.to(self.device)

                optimizer.zero_grad()

                predictions = model(token_ids, output_ids)  # (batch_size, seq_len, vocab_size)

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

                    targets = output_ids  # I was testing some shifting of targets, but it was not helpful
                    predictions = outputs 

                    # Reshape for loss computation
                    predictions = predictions.reshape(-1, predictions.shape[-1])
                    targets = targets.reshape(-1)

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
            # Calculate perplexity for validation set
            val_perplexity = torch.exp(torch.tensor(val_loss)).item()
            train_perplexity = torch.exp(torch.tensor(train_loss)).item()

            # Calculate metrics
            metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity
            })
            print(f'epoch = {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | train_perplexity = {train_perplexity:.3f} | val_perplexity = {val_perplexity:.3f}')


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
    
    def load_model(self, path:str):
        """ Load a pre-trained model from a given path

        **ARGS**
            path: string path to the pre-trained model
        """
        self.to(self.device)
        self.load_state_dict(torch.load(path, weights_only=True))


"""
class TransformerLM(nn.Module):

    Transformer language model

    def __init__(self, vocab_size, embedding_dim, d_model, n_head, n_layer, dropout, pretrained_embeddings=None):
        super().__init__()
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Add projection layer to match dimensions
        self.embedding_projection = nn.Linear(embedding_dim, d_model)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Rest of the initialization remains the same
        self.pos_encoding = LearnedPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = torch.triu(torch.ones(1000, 1000) * float('-inf'), diagonal=1)

    def forward(self, x):  # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = self.embedding_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # Generate causal mask with maximum possible sequence length
        max_seq_len = x.size(1)
        causal_mask = self.causal_mask[:max_seq_len, :max_seq_len]  # Ensure mask matches current sequence length
        causal_mask = causal_mask.to(x.device)
        
        out = self.transformer_encoder(x.transpose(0, 1), mask=causal_mask)  # Transpose for transformer_encoder
        out = out.transpose(0, 1)
        
        # This line gets the vocabulary logits
        out = self.output_projection(out)  # (batch, seq_len, vocab_size)
        
        return out

    def generate(self, x, max_len):
        self.eval()
        generated = x
        for _ in range(max_len):
            with torch.no_grad():
                outputs = self(generated)  # (batch_size, seq_len, vocab_size)
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)  # Get last token
                generated = torch.cat([generated, next_token], dim=1)  # Append to sequence
        return generated

# Move this function definition before it's used (place it before the model initialization)
def load_glove_embeddings(word_to_idx, embedding_dim=100):

    Load GloVe embeddings from gensim

    print("Loading GloVe embeddings from gensim...")
    
    # Load pre-trained GloVe embeddings from gensim
    glove_vectors = load('glove-wiki-gigaword-100')
    
    # Initialize embedding matrix with random values
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_to_idx), embedding_dim))
    
    # Replace random values with GloVe vectors for words that exist
    found_words = 0
    for word, idx in tqdm(word_to_idx.items(), desc="Processing vocabulary"):
        try:
            if word in glove_vectors:
                embedding_matrix[idx] = glove_vectors[word]
                found_words += 1
        except KeyError:
            continue  # Skip words not in GloVe vocabulary
    
    print(f"Found {found_words}/{len(word_to_idx)} words in GloVe")
    return torch.FloatTensor(embedding_matrix)

# Before model initialization, load GloVe embeddings
pretrained_embeddings = load_glove_embeddings(
    train_dataset.encoder_vocab,
    embedding_dim=EMBEDDING_DIM
)

# Update model initialization
model = TransformerLM(
    vocab_size=len(train_dataset.encoder_vocab),
    embedding_dim=EMBEDDING_DIM,
    d_model=DIMENSION,
    n_head=NUM_HEADS,
    n_layer=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=pretrained_embeddings
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.encoder_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


model = model.to(device)

metrics = []
best_f1 = 0
best_val_loss = float('inf')
best_train_loss = float('inf')
# Training Loop
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    for token_ids, output_ids in train_loader:
        token_ids = token_ids.to(device)
        output_ids = output_ids.to(device)

        optimizer.zero_grad()

        predictions = model(token_ids)  # (batch_size, seq_len, vocab_size)

        targets = output_ids # (batch_size, seq_len)

        # Reshape for loss computation
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1)

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
        for token_ids, output_ids in val_loader:
            token_ids = token_ids.to(device)
            output_ids = output_ids.to(device)

            outputs = model(token_ids)

            targets = output_ids  # I was testing some shifting of targets, but it was not helpful
            predictions = outputs 

            # Reshape for loss computation
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1)

            # Compute loss
            loss = loss_fn(predictions, targets)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=-1)
            mask = output_ids != train_dataset.encoder_vocab['<pad>']

            all_predictions.extend(predictions[mask].tolist())
            all_output_ids.extend(output_ids[mask].tolist())

    # compute train and val loss
    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)
    # Calculate perplexity for validation set
    val_perplexity = torch.exp(torch.tensor(val_loss)).item()
    train_perplexity = torch.exp(torch.tensor(train_loss)).item()

    # Calculate metrics
    metrics.append({
    'epoch': epoch + 1,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_perplexity': train_perplexity,
    'val_perplexity': val_perplexity
    })
    print(f'epoch = {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | train_perplexity = {train_perplexity:.3f} | val_perplexity = {val_perplexity:.3f}')

# Get token probabilities for test data
model.eval()
all_batch_perplexity = []
all_sentence_probs = []
total_perplexity = 0
total_tokens = 0

with torch.no_grad():
    for token_ids, output_ids in test_loader:
        token_ids = token_ids.to(device)
        output_ids = output_ids.to(device)

        # Forward pass
        outputs = model(token_ids)  # (batch_size, seq_len, vocab_size)
        
        # Shift logits and output_ids for next-token prediction
        logits = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_len, vocab_size]
        output_ids_flat = output_ids.view(-1)  # [batch_size * seq_len]

        # Apply softmax to get probabilities
        softmax = F.softmax(logits, dim=-1)
        
        # Get probabilities for the correct tokens
        # Make sure output_ids_flat is properly shaped for gathering
        output_ids_flat = output_ids_flat.unsqueeze(-1)  # Add dimension for gathering
        token_probs = softmax.gather(dim=-1, index=output_ids_flat)
        
        # Take log of probabilities
        log_probs = torch.log(token_probs).squeeze(-1)  # Remove gathering dimension
        
        # Create mask for valid tokens
        mask = output_ids_flat.squeeze(-1) != train_dataset.encoder_vocab['<pad>']
        
        # Calculate perplexity for valid tokens
        valid_log_probs = log_probs[mask]
        batch_perplexity = torch.exp(-valid_log_probs.mean()).item()  # Negative because we want negative log likelihood
        
        all_batch_perplexity.append(batch_perplexity)

# Final perplexity
for_csv = pd.DataFrame({'ID': range(len(all_batch_perplexity)), 'ppl': all_batch_perplexity})
for_csv.to_csv(submission, index=False)

# After training loop
train_losses = [m['train_loss'] for m in metrics]
val_losses = [m['val_loss'] for m in metrics]
train_perplexities = [m['train_perplexity'] for m in metrics]
val_perplexities = [m['val_perplexity'] for m in metrics]
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_perplexities, label='Train Perplexity', color='red')
plt.plot(epochs, val_perplexities, label='Val Perplexity', color='green')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig('perplexity_plot.png')
plt.close()
"""