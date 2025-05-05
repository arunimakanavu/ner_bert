import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import ast
from transformers import AutoModelForTokenClassification, AutoTokenizer


df = pd.read_csv('./ner/ner_bio_dataset.csv')


df['labels'] = df['labels'].apply(ast.literal_eval)

label_map = {'B-PER':0, 'I-PER':1, 'B-ORG':2, 'I-ORG':3, 'O':4, 'B-LOC':5, 'I-LOC':6}
df['label_ids'] = df['labels'].apply(lambda label_seq: [label_map[lbl] for lbl in label_seq])



tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']
        label_ids = self.data.iloc[idx]['label_ids']
    
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    # Align labels to the tokenized input
        word_ids = encoding.word_ids(batch_index=0)
        label_aligned = []

# Process each word in the sentence
        for word_id in word_ids:
            if word_id is None:
                label_aligned.append(-100)  # Ignore token for loss
            elif word_id >= len(label_ids):
                label_aligned.append(-100)    #safeguard for subwords that exceed label length
            else:
                label_aligned.append(label_ids[word_id])


        return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'labels': torch.tensor(label_aligned, dtype=torch.long)
        }



train_dataset = SentimentDataset(train_data, tokenizer, max_length=128)
test_dataset = SentimentDataset(test_data, tokenizer, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=7)

model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs=20
for epoch in range(epochs):
    epoch_loss=0
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        optimizer.zero_grad()

        outputs=model(input_ids, attention_mask=attention_mask, labels=labels)
        loss=outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss/len(train_dataloader)}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, predicted = torch.max(logits, dim=2)

        mask = labels != -100
        correct += ((predicted == labels) & mask).sum().item()
        total += mask.sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


model.save_pretrained('./ner/ner_transformer_model')
tokenizer.save_pretrained('./ner/ner_transformer_model')


for batch in test_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    for i in range(input_ids.size(0)):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        pred_labels = predictions[i]
        true_labels = labels[i]

        print("\nTokens:", input_tokens)
        print("Predicted:", pred_labels.tolist())
        print("Actual:   ", true_labels.tolist())
        break
    break

