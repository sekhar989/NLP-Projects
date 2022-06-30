# Data

- `words.txt` and `tags.txt` $\rightarrow$ used to create the vocab and ner-tag maps
- `train`, `test` and `val` folders each containing two files. For replication purposes.
    - `sentences.txt` $\rightarrow$ sentences in each line
    - `labels.txt` $\rightarrow$ corresponding ner-tags of each sentences
    
**Note: As the dataset is already partitioned the dataset class that will manage the data operations can directly read the data and address the necessary modifications.**

## Vocab & Tag Maps


```python
words_path = '../data/words.txt'
tags_path = '../data/tags.txt'
```


```python
vocab = {}
with open(words_path) as f:
    for i, l in enumerate(f.read().splitlines()):
        vocab[l] = i
vocab['<PAD>'] = len(vocab)

print("Thousands: \t", vocab['Thousands'], "\tUknown:\t", vocab['UNK'], "\tPAD\t", vocab['<PAD>'])
print()
c = 0
for i in vocab.items():
    print(i)
    print('.'*50)
    c += 1
    if c > 5: break
```

    Thousands: 	 0 	Uknown:	 35179 	PAD	 35180
    
    ('Thousands', 0)
    ..................................................
    ('of', 1)
    ..................................................
    ('demonstrators', 2)
    ..................................................
    ('have', 3)
    ..................................................
    ('marched', 4)
    ..................................................
    ('through', 5)
    ..................................................


- A `<PAD>` token is set which will be used to pad the sentences upto the maximum length of the sequences
- An `UNK` token is added to address out of vocabulary words/tokens


```python
tag_map = {}
with open(tags_path) as f:
    for i, t in enumerate(f.read().splitlines()):
        tag_map[t] = i

tag_map
```




    {'O': 0,
     'B-geo': 1,
     'B-gpe': 2,
     'B-per': 3,
     'I-geo': 4,
     'B-org': 5,
     'I-org': 6,
     'B-tim': 7,
     'B-art': 8,
     'I-art': 9,
     'I-per': 10,
     'I-gpe': 11,
     'I-tim': 12,
     'B-nat': 13,
     'B-eve': 14,
     'I-eve': 15,
     'I-nat': 16}



## Loading Training data and performing Transformations
- Same will be applicable for both test and validation sets


```python
# Load Data in lists
with open("../data/train/sentences.txt", "r") as f:
    sentences = f.read().splitlines()
    
for i in sentences[:5]:
    print(i)
    print('.' * 100)
```

    Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country .
    ....................................................................................................
    Families of soldiers killed in the conflict joined the protesters who carried banners with such slogans as " Bush Number One Terrorist " and " Stop the Bombings . "
    ....................................................................................................
    They marched from the Houses of Parliament to a rally in Hyde Park .
    ....................................................................................................
    Police put the number of marchers at 10,000 while organizers claimed it was 1,00,000 .
    ....................................................................................................
    The protest comes on the eve of the annual conference of Britain 's ruling Labor Party in the southern English seaside resort of Brighton .
    ....................................................................................................



```python
# Load Data in lists
with open("../data/train/labels.txt", "r") as f:
    labels = f.read().splitlines()
    
for i in labels[:5]:
    print(i)
    print('.' * 100)
```

    O O O O O O B-geo O O O O O B-geo O O O O O B-gpe O O O O O
    ....................................................................................................
    O O O O O O O O O O O O O O O O O O B-per O O O O O O O O O O O
    ....................................................................................................
    O O O O O O O O O O O B-geo I-geo O
    ....................................................................................................
    O O O O O O O O O O O O O O O
    ....................................................................................................
    O O O O O O O O O O O B-geo O O B-org I-org O O O B-gpe O O O B-geo O
    ....................................................................................................


## Data Conversion
- Convert sentences into list of id's based on the word2id $\rightarrow$ vocab mapping
- Convert tags in to list of tag id's based on the tag2id $\rightarrow$ tags mapping


```python
from typing import List, Dict
import random
random.seed(518123)
```


```python
def convert_sentence2id(sentence: str, vocab_map: Dict) -> List:
    sentence_id = []
    for token in sentence.split(' '):
        if token in vocab_map:
            sentence_id.append(vocab_map[token])
        else:
            sentence_id.append(vocab_map['UNK'])
    return sentence_id            
```


```python
def convert_tags2id(tag_list: str, tag_map: Dict) -> List:
    tag_id = []
    for label in tag_list.split(' '):
        tag_id.append(tag_map[label])
    return tag_id
```


```python
rand_id = random.choice(range(len(sentences)))
rand_sent = sentences[rand_id]
print(rand_sent)
print('.'*len(rand_sent))
print(convert_sentence2id(rand_sent, vocab_map=vocab))
print()
rand_labels = labels[rand_id]
print(rand_labels)
print('.'*len(rand_labels))
print(convert_tags2id(rand_labels, tag_map=tag_map))
```

    In a statement Monday , Mr. Peres said " there exists no basis in reality for the claims published " by the British newspaper , The Guardian .
    ..............................................................................................................................................
    [345, 45, 1171, 1564, 93, 816, 8887, 172, 35, 596, 10871, 388, 2051, 11, 7814, 223, 9, 2865, 2573, 35, 191, 9, 16, 1765, 93, 61, 2646, 21]
    
    O O O B-tim O B-per I-per O O O O O O O O O O O O O O O B-gpe O O B-org I-org O
    ...............................................................................
    [0, 0, 0, 7, 0, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 5, 6, 0]



```python
assert len(sentences) == len(labels)
for s, l in zip(sentences, labels):
    try:
        assert len(s.split(' ')) == len(l.split(' '))
    except AssertionError:
        print(s, l)
        continue
```

## Dataclass
- Pass the lists into torch dataset/data-generator class


```python
train_sentence = "../data/train/sentences.txt"
train_labels = "../data/train/labels.txt"

valid_sentence = "../data/val/sentences.txt"
valid_labels = "../data/val/labels.txt"

test_sentence = "../data/test/sentences.txt"
test_labels = "../data/test/labels.txt"
```


```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
```


```python
class ner_data(Dataset):
    
    def __init__(self, sentences_path: str, labels_path: str, vocab_map: Dict, tags_map: Dict) -> None:
        super().__init__()
        
        with open(sentences_path, "r") as f:
            self.sentences = f.read().splitlines()
        
        with open(labels_path, "r") as f:
            self.labels = f.read().splitlines()
            
        self.max_len = max([len(sentence) for sentence in self.sentences])
        self.vocab_map = vocab_map
        self.tags_map = tags_map
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sentence_padded = np.array(self.max_len * [vocab["<PAD>"]])
        labels_padded = np.array(self.max_len * [vocab["<PAD>"]])
        
        sentence = convert_sentence2id(self.sentences[idx], self.vocab_map)
        labels = convert_tags2id(self.labels[idx], self.tags_map)
        
        assert len(sentence) == len(labels)
        
        sentence_padded[:len(sentence)] = sentence
        labels_padded[:len(labels)] = labels
        
        return torch.tensor(sentence_padded, dtype=torch.long), torch.tensor(labels_padded, dtype=torch.long)
```


```python
ner_train = ner_data(sentences_path=train_sentence,
                     labels_path=train_labels,
                     vocab_map=vocab,
                     tags_map=tag_map)
```


```python
ner_train.max_len, len(ner_train)
```




    (541, 33570)




```python
for i in range(10):
    data = ner_train.__getitem__(i)
    print(data[0].shape, data[1].shape)
```

    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])
    torch.Size([541]) torch.Size([541])



```python
batch_size = 128
train_loader = DataLoader(dataset=ner_train, batch_size=batch_size, shuffle=True, num_workers=4)
```


```python
tl = iter(train_loader)
```


```python
data_batch = next(tl)
```


```python
data_batch[0].shape, data_batch[1].shape
```




    (torch.Size([128, 541]), torch.Size([128, 541]))




```python
def create_ner_data_loader(sentences_path, labels_path, vocab_map, tags_map,
                           batch_size=8, shuffle=True, num_workers=4):
    
    ner = ner_data(sentences_path, labels_path, vocab_map, tags_map)
    
    return DataLoader(dataset=ner, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
```


```python
train_loader = create_ner_data_loader(train_sentence, train_labels, vocab, tag_map, batch_size=batch_size)
valid_loader = create_ner_data_loader(valid_sentence, valid_labels, vocab, tag_map, batch_size=batch_size)
test_loader = create_ner_data_loader(test_sentence, test_labels, vocab, tag_map, batch_size=batch_size)
```


```python
vl = iter(valid_loader)
valid_data = next(vl)
```


```python
valid_data[0].shape, valid_data[1].shape
```




    (torch.Size([128, 424]), torch.Size([128, 424]))



## Model


```python
from torch import nn
import torch.nn.functional as F

class NER_Tagger(nn.Module):
    
    def __init__(self, vocab_size: int, embedding_size: int, dense_size: int, device: str = 'cpu') -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size//2, batch_first=True)
        self.dense = nn.Linear(embedding_size//2, dense_size)
        
        self.to(device)

    def forward(self, X):
        """
        processed --> will have 3 outputs
            - hidden states for each input sequence
            - final hidden state for each element in the sequence
            - final cell state for each element in the sequence
            
        processed[0].shape = (batch_size, sequence_length, h_out_size)
        cessed[1][0].shape = (1, batch_size, h_out_size)
        processed[1][1].shape = (1, batch_size, h_out_size)
        """
        
        embedded = self.embedding(X)
        processed = self.lstm(embedded)
        processed = self.dense(processed[0])
        return F.log_softmax(processed, dim=-1)
```


```python
tagger = NER_Tagger(len(vocab), 50, len(tag_map))
tagger
```




    NER_Tagger(
      (embedding): Embedding(35181, 50)
      (lstm): LSTM(50, 25, batch_first=True)
      (dense): Linear(in_features=25, out_features=17, bias=True)
    )




```python
predicted_tags = tagger(data_batch[0])
predicted_tags
```




    tensor([[[-2.9543, -2.6421, -2.7909,  ..., -2.7396, -3.0655, -2.9513],
             [-2.7502, -2.7699, -2.8578,  ..., -2.6967, -3.0108, -2.9649],
             [-2.7003, -2.8158, -2.8555,  ..., -2.7339, -3.0042, -3.0388],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]],
    
            [[-2.8164, -2.6799, -2.9589,  ..., -2.8014, -2.8455, -2.9777],
             [-2.9139, -2.6669, -2.7430,  ..., -2.7715, -3.0560, -2.8741],
             [-2.8421, -2.7951, -2.8769,  ..., -2.8197, -2.9766, -2.9123],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]],
    
            [[-2.7109, -2.6404, -2.9564,  ..., -2.8495, -3.0234, -2.9504],
             [-2.8607, -2.6226, -2.9124,  ..., -2.8620, -3.0945, -2.9324],
             [-2.7269, -2.8098, -2.7958,  ..., -2.8594, -3.1460, -2.8824],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]],
    
            ...,
    
            [[-2.6590, -2.7491, -2.8446,  ..., -2.9621, -3.0982, -2.9438],
             [-2.6916, -2.6842, -2.9594,  ..., -2.7870, -2.9410, -3.0102],
             [-2.9238, -2.8172, -2.8601,  ..., -2.7522, -2.9630, -2.8569],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]],
    
            [[-2.7591, -2.7021, -2.8990,  ..., -2.6591, -3.1008, -3.0319],
             [-2.7096, -2.6931, -2.9067,  ..., -2.7982, -2.9856, -3.0354],
             [-2.7242, -2.7015, -2.8836,  ..., -2.8020, -3.0177, -2.9704],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]],
    
            [[-2.8857, -2.7769, -2.8912,  ..., -2.7361, -3.0308, -2.9395],
             [-2.8476, -2.8471, -2.8890,  ..., -2.8549, -3.0612, -2.8391],
             [-2.8937, -2.7629, -2.7980,  ..., -2.8127, -3.0623, -2.8047],
             ...,
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115],
             [-2.8359, -2.4105, -3.0856,  ..., -2.7355, -2.9659, -3.0115]]],
           grad_fn=<LogSoftmaxBackward0>)




```python
predicted_tags.shape, predicted_tags[0].shape
```




    (torch.Size([128, 541, 17]), torch.Size([541, 17]))




```python
torch.argmax(predicted_tags, dim=-1).shape
```




    torch.Size([128, 541])




```python
data_batch[1]
```




    tensor([[    0,     1,     0,  ..., 35180, 35180, 35180],
            [    1,     0,     0,  ..., 35180, 35180, 35180],
            [    0,     0,     0,  ..., 35180, 35180, 35180],
            ...,
            [    5,     0,     0,  ..., 35180, 35180, 35180],
            [    0,     0,     0,  ..., 35180, 35180, 35180],
            [    3,    10,     0,  ..., 35180, 35180, 35180]])




```python
(torch.argmax(predicted_tags, dim=-1) == data_batch[1])
```




    tensor([[False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            ...,
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False]])




```python
pad_mask = data_batch[1] != vocab['<PAD>']
pad_mask
```




    tensor([[ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            ...,
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False]])




```python
((torch.argmax(predicted_tags, dim=-1) == data_batch[1]) * pad_mask).sum(), pad_mask.sum(), ((torch.argmax(predicted_tags, dim=-1) == data_batch[1]) * pad_mask).sum() / pad_mask.sum()
```




    (tensor(92), tensor(2825), tensor(0.0326))




```python
data_batch[1] * pad_mask
```




    tensor([[ 0,  1,  0,  ...,  0,  0,  0],
            [ 1,  0,  0,  ...,  0,  0,  0],
            [ 0,  0,  0,  ...,  0,  0,  0],
            ...,
            [ 5,  0,  0,  ...,  0,  0,  0],
            [ 0,  0,  0,  ...,  0,  0,  0],
            [ 3, 10,  0,  ...,  0,  0,  0]])




```python
type(data_batch[0])
```




    torch.Tensor



## Check to see the operations on dimensions


```python
X = torch.rand((2, 4, 3))
X
```




    tensor([[[0.0565, 0.6274, 0.7453],
             [0.9940, 0.0367, 0.6191],
             [0.2324, 0.2376, 0.1080],
             [0.5215, 0.6242, 0.0741]],
    
            [[0.6833, 0.1395, 0.2892],
             [0.6755, 0.9305, 0.8443],
             [0.7468, 0.9332, 0.3232],
             [0.8526, 0.5568, 0.4798]]])




```python
test_log_softmax = F.log_softmax(X, dim=-1)
test_log_softmax
```




    tensor([[[-1.5605, -0.9896, -0.8717],
             [-0.7282, -1.6855, -1.1030],
             [-1.0607, -1.0555, -1.1850],
             [-1.0107, -0.9079, -1.4581]],
    
            [[-0.8131, -1.3569, -1.2071],
             [-1.2454, -0.9905, -1.0766],
             [-1.0507, -0.8643, -1.4743],
             [-0.8890, -1.1848, -1.2618]]])




```python
torch.exp(test_log_softmax).sum(dim=-1)
```




    tensor([[1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000]])



## Cost Function and Optimizers

### Device Setup, Learning Rate


```python
from tqdm.notebook import tqdm

for i in tqdm(range(1000), dynamic_ncols=True):
    pass
```


      0%|                                                                                                         …



```python
def prediction_evaluation(predictions, true_labels, pad_value):
    """
    Inputs:
        pred: prediction array with shape -> (num_examples, max sentence length in batch)
        labels: array of size (batch_size, seq_len)
        pad: integer representing pad character
    Outputs:
        accuracy: float
    """
    # Create mask matrix equal to the shape of the true-labels matrix
    pad_mask = true_labels != pad_value
    
    # Calculate Accuracy
    accuracy = ((predictions == true_labels) * pad_mask).sum() / pad_mask.sum()
    
    return accuracy
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 1e-2
lr_step_size = 2           # every 2 epochs
lr_reduction_pc = 0.95     # 95% reduction of lr

grad_clip_threshold = 0.3

epochs = 5

tagger = NER_Tagger(vocab_size=len(vocab), embedding_size=50, dense_size=len(tag_map), device=device)
```


```python
# Loss/Cost Function
loss_function = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>']).to(device)

# Optimizer --> model parameters and learning rate
# Class weights can also be specified if necessary
optimizer = torch.optim.Adam(params=tagger.parameters(), lr=lr)

# learning rate scheduler --> slow reduction of learning rate from a higher value
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=lr_reduction_pc)

# dl tricks
#    gradient clippings --> to avoid vanishing gradient or gradient explosion
gradient_clip = nn.utils.clip_grad_value_
gradient_clip(parameters=tagger.parameters(), clip_value=grad_clip_threshold)
```


```python
validation_metric_tracker = []
steps = 0

# Epoch Loop --> 1 epoch is one forward pass through all the training data
for epoch in tqdm(range(epochs), dynamic_ncols=True, desc='Epoch Progress'):

    # set model to training mode
    tagger.train()
    epoch_loss, train_predictions = [], []

    # training loop for each mini-batch
    for sentences, true_tags in tqdm(train_loader, dynamic_ncols=True, desc='Training Progress'):
        # set grads to zero
        optimizer.zero_grad()

        # forward pass
        sentences, true_tags = sentences.to(device), true_tags.to(device)
        predicted_tags = tagger(sentences)

        # loss calculation
        loss = loss_function(predicted_tags.view(-1, predicted_tags.shape[2]),
                             true_tags.view(-1))

        # Track Loss for each epoch
        epoch_loss.append(loss.item())

        # calculate training metrics
        train_accuracy = prediction_evaluation(torch.argmax(predicted_tags, dim=-1), true_tags, pad_value=vocab['<PAD>'])
        train_predictions.append(train_accuracy.item())

        # backward pass i.e. gradient calculations
        loss.backward()

        # optimizer step
        optimizer.step()

        # increment training steps
        steps += 1

    print('Training {:>4d} steps  --- Accuracy {:>5.4f}  |  Epoch Loss  {:>5.4f}'.format(steps, np.mean(train_predictions),
                                                                                     np.mean(epoch_loss)))
    
    # validation loop after each training epoch
    # set no grad
    with torch.no_grad():
        tagger.eval()
        validation_predictions = []
        
        # perform validation
        # validation loop for each mini-batch
        for val_sentences, val_true_tags in tqdm(train_loader, dynamic_ncols=True, desc='Validation Progress'):
            
            val_sentences, val_true_tags = val_sentences.to(device), val_true_tags.to(device)
            predicted_tags = tagger(val_sentences)
            
            # calculate validation metrics
            validation_accuracy = prediction_evaluation(torch.argmax(predicted_tags, dim=-1), val_true_tags, pad_value=vocab['<PAD>'])
            validation_predictions.append(validation_accuracy.item())
        
        print('Validation Accuracy {:>5.4f}'.format(np.mean(validation_predictions)))
    
    lr_scheduler.step()

# save checkpoint (conditioned on performance or time-steps)
```
    Training 1315 steps  --- Accuracy 0.9726  |  Epoch Loss  0.0867   |    Validation Accuracy 0.9759



```python
validator = iter(test_loader)
```


```python
x_test, y_test = validator.next()
```


```python
x_test = x_test.to('cuda')
predicted_test = tagger(x_test)
```


```python
prediction_evaluation(torch.argmax(predicted_test, dim=-1).detach().cpu(), y_test, pad_value=vocab['<PAD>'])
```




    tensor(0.9537)




```python
pred_tags = torch.argmax(predicted_test, dim=-1).detach().cpu()
```


```python
mask = y_test[0] != vocab['<PAD>']
```


```python
reverse_tag_map = {v:k for k, v in tag_map.items()}
reverse_tag_map
```




    {0: 'O',
     1: 'B-geo',
     2: 'B-gpe',
     3: 'B-per',
     4: 'I-geo',
     5: 'B-org',
     6: 'I-org',
     7: 'B-tim',
     8: 'B-art',
     9: 'I-art',
     10: 'I-per',
     11: 'I-gpe',
     12: 'I-tim',
     13: 'B-nat',
     14: 'B-eve',
     15: 'I-eve',
     16: 'I-nat'}




```python
reverse_vocab_map = {v:k for k, v in vocab.items()}
```

```python
red = "\033[1;31m"
green = "\033[1;32m"
purple= "\033[1;35m"
reset= "\033[0m"
```


```python
sentence = [reverse_vocab_map[i.item()] for i in x_test[0][mask].detach().cpu()]
pred_tags_converted = [reverse_tag_map[i.item()] for i in pred_tags[0][mask]]
true_tags_converted = [reverse_tag_map[i.item()] for i in y_test[0][mask]]

print("{:>20} | {:>7} | {:>5}".format("Token", "Pred", "True NER Tag"))
print('-'*70)
for s, p, t in zip(sentence, pred_tags_converted, true_tags_converted):
    if p == t:
        p = green + p + reset
    else:
        p = red + p + reset
    t = purple + t + reset
    print("{:>20} | {:>17}  | {:>15}".format(s, p, t))
```

|               Token |    Pred | True NER Tag|
|---|---|---|
|Somali |  B-gpe  | B-org|
|officials |      O  |    O|
|say |      O  |    O|
|no |      O  |    O|
|one |      O  |    O|
|was |      O  |    O|
|injured |      O  |    O|
|by |      O  |    O|
|the |      O  |    O|
|blast |      O  |    O|
|targeting |      O  |    O|
|Mayor |  B-per  | B-per|
|Mohamed |  I-per  | I-per|
|Dheere |  I-per  | I-per|
|in |      O  |    O|
|the |      O  |    O|
|city |      O  |    O|
|'s |      O  |    O|
|northern |      O  |    O|
|Shibis |  B-per  | B-geo||
|neighborhood |      O  |    O|
|. |      O  |    O|

```python
random_id = random.randint(0, len(test_loader))
```


```python
sentence = [reverse_vocab_map[i.item()] for i in x_test[random_id][mask].detach().cpu()]
pred_tags_converted = [reverse_tag_map[i.item()] for i in pred_tags[random_id][mask]]
true_tags_converted = [reverse_tag_map[i.item()] for i in y_test[random_id][mask]]

print("{:>20} | {:>7} | {:>5}".format("Token", "Pred", "True Tag"))
print('-'*70)
for s, p, t in zip(sentence, pred_tags_converted, true_tags_converted):
    if p == t:
        p = green + p + reset
    else:
        p = red + p + reset
    t = purple + t + reset
    print("{:>20} | {:>17}  | {:>15}".format(s, p, t))
```

|   Token |    Pred | True Tag |
|---|---|---|
|Alassane |  B-org  | B-per|
|Ouattara |  I-per  | I-per|
|arrived |      O  |    O|
|in |      O  |    O|
|Abidjan |  B-geo  | B-geo|
|Wednesday |  B-tim  | B-tim|
|, |      O  |    O|
|greeted |      O  |    O|
|by |      O  |    O|
|a |      O  |    O|
|small |      O  |    O|
|group |      O  |    O|
|of |      O  |    O|
|supporters |      O  |    O|
|and |      O  |    O|
|dozens |      O  |    O|
|of |      O  |    O|
|U.N. |  B-org  | B-geo|
|peacekeepers |      O  |    O|
|who |      O  |    O|
|will |O|    O|
|provide |      O  |    O|

```python

```
