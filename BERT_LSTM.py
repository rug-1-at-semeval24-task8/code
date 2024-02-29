import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



import argparse

parser = argparse.ArgumentParser(description='BERT embedding with LSTM Model')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default =20, type=int, help="number of epochs")
parser.add_argument('--classes', default =2, type=int, help="number of classes")
parser.add_argument('--batch_size', default =32, type=int)
parser.add_argument('--task', default='A', type=str, help='specify Task A or B')
parser.add_argument('--multilingual', default=False, help="Specify either monolingual or multilingual")
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
parser.add_argument('--experiment_name', default='v1', type=str, help="specify a name for your experiment to name the saved files")
args = parser.parse_args()

if args.multilingual== True: 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states = True)
else:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

if args.task == 'A':
    labels= [0,1]
    idx2labels = {0:'human text', 1:'machine text'}
else:
    labels= [0,1,2,3,4,5]
    idx2labels = {0:'human', 1:'chatGPT', 2:'cohere', 3:'davinci', 4:'bloomz', 5:'dolly'}

def BERT_EMBEDDING(texts):
    sentence_embedding= []
    bert_model
    bert_model.eval()
    for text in texts:
        tokenized_text = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        try: 
            outputs = bert_model(tokens_tensor, segments_tensors)
        except: 
            print(f"tokens: {tokens_tensor.shape}, segments: {segments_tensors.shape}")
        hidden_states = outputs[2]
        # `token_vecs` is a tensor with shape [len(text) x 768]
        token_vecs = hidden_states[-4][0]
        # Calculate the average of all (len(text)) token vectors.
        embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding.append(embedding)
    del tokenized_text, indexed_tokens, segments_ids, tokens_tensor, segments_tensors, outputs, token_vecs, hidden_states, embedding 
    return torch.stack(sentence_embedding, dim=0).to(args.device)

def load_dataset():
    if args.task=='B':
        train_path = '../dataset/SubtaskB/subtaskB_train.jsonl'
        dev_path = '../dataset/SubtaskB/subtaskB_dev.jsonl'
        test_path = '../dataset/test_data/subtaskB.jsonl'
    elif args.multilingual==False: 
        train_path = '../dataset/SubtaskA/subtaskA_train_monolingual.jsonl'
        dev_path = '../dataset/SubtaskA/subtaskA_dev_monolingual.jsonl'
        test_path = '../dataset/test_data/subtaskA_monolingual.jsonl'
    else:
        train_path = '../dataset/SubtaskA/subtaskA_train_multilingual.jsonl'
        dev_path = '../dataset/SubtaskA/subtaskA_dev_multilingual.jsonl'
        test_path = '../dataset/test_data/subtaskA_multilingual.jsonl'

    train_df = pd.read_json(path_or_buf=train_path, lines=True)
    test_df = pd.read_json(path_or_buf=test_path, lines=True)
    #return train_df['text'], train_df['label'], test_df['text'], test_df['label']
    return train_df['text'], train_df['label'], test_df['text'], test_df['id']

test_flag= False

class customDataset(Dataset):
    global test_flag
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        if test_flag==False:
            self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        sample = {}
        sample['encoding'] = self.embeddings[idx]
        if test_flag ==False:
            sample['label'] = self.labels[idx]
        return sample


class LSTM_MODEL(nn.Module):
    def __init__(self, num_class=args.classes):  # num_class set to 2 for task A
        super(LSTM_MODEL, self).__init__()
        print(args.classes)
        self.lstm = nn.LSTM(input_size = 768,
                            hidden_size =128, 
                            num_layers =2, 
                            dropout =0.2,
                            batch_first = True)
        self.fc = nn.Linear(128, num_class)
    
    def forward(self, embedding):
        output, hidden = self.lstm(embedding)
        return self.fc(output)

def train_and_test(model,
          device =args.device):

      #train_X, train_y, test_X, test_y = load_dataset()
      train_X, train_y, test_X, test_id = load_dataset()
      #-------- REMOVE THE FOLLOWING FOUR LINES TO RUN ON FULL DATASET------
      train_X=train_X[:1000]
      train_y=train_y[:1000]
      test_X = test_X[:100]
      test_id = test_id[:100]
      #test_y = test_y[:100]
      train_X_embedding = BERT_EMBEDDING(train_X)
      
      

      torch.manual_seed=42
      torch.cuda.manual_seed =42

      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(params= model.parameters(), lr=args.lr)

      train_y = torch.Tensor(train_y).type(torch.LongTensor)
      #test_y = torch.Tensor(test_y).type(torch.LongTensor)
      train_y = train_y.to(args.device)
      #test_y = test_y.to(args.device)
      train_dataset = customDataset(train_X_embedding, train_y)
      dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
      train_loss_per_epoch = []
      #print(train_X_embedding.shape)
      #print(test_X_embedding.shape)

      model.to(device)
      for epoch in range(args.epochs):
            train_loss =0.0
            model.train()
            for batch in dataloader:
                  X= batch['encoding']
                  y = batch['label']
                  y_logits = model(X)
                  optimizer.zero_grad()
                  loss = loss_fn(y_logits, y)
                  train_loss += loss 
                  loss.backward(retain_graph=True)
                  optimizer.step()
            print(f"Epoch {epoch}: Loss {train_loss}")
            train_loss = (train_loss/len(train_y))*100 
            train_loss_per_epoch.append(train_loss)
      model_path = '../saved_models/saved_model_'+args.experiment_name
      torch.save(model.state_dict(), model_path)
      global test_flag 
      test_flag=True
      del train_X_embedding
      test_X_embedding = BERT_EMBEDDING(test_X)
      with torch.inference_mode():
        test_dataset = customDataset(test_X_embedding)
        dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)
        y_pred = np.array([])
        #test_loss = 0.0
        for batch in  dataloader:
            X= batch['encoding']
            #y = batch['label']
            y_logits = model(X)
            #loss = loss_fn(y_logits, y)
            #test_loss += loss
            batch_pred = y_logits.argmax(dim=1).detach().cpu().numpy()
            y_pred = np.append(y_pred, batch_pred)
        #test_loss = (test_loss/len(y_pred))*100
        y_pred = y_pred.astype(int)
        df = pd.DataFrame({'id': test_id, 'label': y_pred}, columns=['id', 'label'] )
        df.to_csv( '../output_files/'+args.experiment_name+'_test_label.csv', index=False )

if __name__ == "__main__":
    model = LSTM_MODEL(num_class=args.classes)
    train_and_test(model=model)
