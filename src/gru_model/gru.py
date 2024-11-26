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


path = os.path.join(os.path.dirname(__file__), '../../data/coNER_data/')

def retrieve_coNER(file: str):
    """
    Retrieves tokens and target IOB ner_tags from coNER file
    """

    try:
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        tokens = [item['tokens'] for item in data]
        ner_tags = [item['ner_tags'] for item in data]

        return tokens, ner_tags
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return [], []
    except KeyError as e:
        print(f"Error: Missing key in JSON data - {e}")
        return [], []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return [], []

torch.manual_seed(57)

source_train, target_train = retrieve_coNER("en_coNER_train.json")
source_val, target_val = retrieve_coNER("en_coNER_validation.json")

x = [source_train[i] for i in range(len(source_train))]
x_split_train = []
for i in range(len(x)):
    x_split_train.append(x[i])
y = [target_train[i] for i in range(len(target_train))]
y_split_train = []
for i in range(len(y)):
    y_split_train.append(y[i])

x_val = [source_val[i] for i in range(len(source_val))]
x_split_valtest = []
for i in range(len(x_val)):
    x_split_valtest.append(x_val[i])
y_val = [target_val[i] for i in range(len(target_val))]
y_split_valtest = []
for i in range(len(y_val)):
    y_split_valtest.append(y_val[i])


# train test split
x_split_val, x_split_test, y_split_val, y_split_test = sklearn.model_selection.train_test_split(x_split_valtest, y_split_valtest, test_size=0.50, random_state=64)

class TagingDataset(Dataset):
    def __init__(self, x, y, token_vocab=None, tag_vocab=None, training=True):
        # Create vocabularies if training
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}

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
            tag_ids = [self.tag_vocab[tag] for tag in y[i]]
            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_tag_ids.append(torch.tensor(tag_ids))

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]

# create datasets
train_dataset = TagingDataset(x_split_train, y_split_train, training=True)
val_dataset = TagingDataset(x_split_val, y_split_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
test_dataset = TagingDataset(x_split_test, y_split_test, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)



# collate token_ids and tag_ids to make mini-batches
def collate_fn(batch):
    # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]

    # Separate sentences and tags
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]

    # Pad sequences
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab['<PAD>'])
    # sentences_pad.size()  (batch_size, seq_len)
    tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab['<PAD>'])
    # tags_pad.size()  (batch_size, seq_len)
    return sentences_padded, tags_padded

#model definition
class GRUModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        out, _ = self.gru(embeddings)
        out = self.fc(out)
        return out


EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, shuffle=False)

model = GRUModel(
    vocab_size=len(train_dataset.token_vocab),
    tagset_size=len(train_dataset.tag_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    for token_ids, tag_ids in train_loader:
        token_ids = token_ids.to(device)
        tag_ids = tag_ids.to(device)

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
        for token_ids, tag_ids in val_loader:
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)

            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            outputs = outputs.view(-1, outputs.shape[-1])
            tag_ids = tag_ids.view(-1)
            loss = loss_fn(outputs, tag_ids)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            mask = tag_ids != train_dataset.tag_vocab['<PAD>']

            all_predictions.extend(predictions[mask].tolist())
            all_tags.extend(tag_ids[mask].tolist())

    # compute train and val loss
    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)

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
"""
# plot training metrics
line_chart = plt.figure()
x_axis = [x[0] for x in metrics]
y_axis = [x[1] for x in metrics]
y_axis2 = [x[2] for x in metrics]
y_axis3 = [x[3] for x in metrics]
plt.plot(x_axis, y_axis, label="Training Loss", color="blue", linestyle="-.")
plt.plot(x_axis, y_axis2, label="Validation Loss", color="orange", linestyle="-.")
plt.plot(x_axis, y_axis3, label="F1 Score", color="green", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss/F1 Score")
plt.legend()
plt.title("Training Metrics")
plt.savefig("training_metrics.png")
"""
best_model = GRUModel(
    vocab_size=len(train_dataset.token_vocab),
    tagset_size=len(train_dataset.tag_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

best_model.load_state_dict(torch.load("best_model.pt", weights_only=True))

# Input testing data to create predictions
def Predict(model):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for token_ids, _ in test_loader:
            token_ids = token_ids.to(device)
            outputs = model(token_ids) # (batch_size, seq_len, tagset_size)
            predictions = outputs.argmax(dim=-1)  # Get predictions for each token
            all_predictions.extend(predictions.view(-1).tolist())
    return all_predictions

predictions = Predict(best_model)

# Convert numeric predictions back to tags
for tag_ids in train_dataset.tag_vocab:
    for i in range(len(predictions)):
        if predictions[i] == train_dataset.tag_vocab[tag_ids]:
            predictions[i] = tag_ids

# Calculate lengths of each test sequence
x_split_test_sizes = [len(seq) for seq in x_split_test]

# Split predictions according to original sequence lengths
predictions_split = []
for i in range(len(x_split_test)):
    predictions_split.append(predictions[sum(x_split_test_sizes[:i]):sum(x_split_test_sizes[:i+1])])

output = []
for i in range(len(predictions_split)):
    output.append([i+1," ".join(predictions_split[i])])

for_csv = pd.DataFrame(output, columns = ["ID", "IOB Slot tags"])
for_csv.to_csv("gru_predictions.csv", index = False)