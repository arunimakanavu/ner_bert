
# 🧠 Named Entity Recognition (NER) with BERT

This project demonstrates how to fine-tune a BERT model for Named Entity Recognition (NER) using Hugging Face Transformers and PyTorch. The model is trained to recognize named entities such as people, organizations, and locations in text using a BIO tagging scheme.

---

## 📂 Project Structure

```
ner/
├── ner_bio_dataset.csv      # Training dataset (BIO-tagged)
├── ner_eval.json            # Evaluation dataset with entities and labels
├── ner_transformer_model/   # Saved model and tokenizer directory
```

---

## 📊 Dataset Description

- **Format**: The training data is in CSV format with fields:
  - `sentence`: A raw text sentence.
  - `labels`: A list of NER tags corresponding to each word in the sentence.

- **Label Scheme**: BIO tagging format.
  - `B-PER`, `I-PER`: Beginning and inside of a person entity
  - `B-ORG`, `I-ORG`: Beginning and inside of an organization
  - `B-LOC`, `I-LOC`: Beginning and inside of a location
  - `O`: Non-entity tokens

---

## 🏗️ Preprocessing Pipeline

1. Convert string labels to lists using `ast.literal_eval`.
2. Map tag strings to numerical IDs using a custom `label_map`.
3. Tokenize input using `BertTokenizerFast` with `is_split_into_words=True`.
4. Align word-level labels with subword tokens, using `-100` for ignored positions.

```python
label_map = {
    'B-PER': 0, 'I-PER': 1,
    'B-ORG': 2, 'I-ORG': 3,
    'O': 4, 'B-LOC': 5, 'I-LOC': 6
}
```

---

## 🧾 Dataset Loader

A custom PyTorch `Dataset` class (`SentimentDataset`) prepares tokenized inputs and aligns token-level labels. It returns a dictionary containing:

- `input_ids`
- `attention_mask`
- `labels`

---

## 🧠 Model Training

- **Model**: `BertForTokenClassification` with 7 output labels.
- **Loss Function**: Cross-entropy, ignoring tokens with label `-100`.
- **Optimizer**: AdamW
- **Epochs**: 20
- **Batch Size**: 8

### 🔁 Training Loop

```python
for epoch in range(epochs):
    for batch in train_dataloader:
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        ...
```

### ✅ Evaluation

During evaluation, tokens labeled with `-100` are excluded. Accuracy is computed as the ratio of correctly predicted valid tokens.

```python
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 💾 Saving the Model

The model and tokenizer are saved to disk for later inference:

```python
model.save_pretrained('./ner/ner_transformer_model')
tokenizer.save_pretrained('./ner/ner_transformer_model')
```

---

## 🔍 Evaluation on JSON Entity Data

A separate evaluation phase is performed on a JSON-formatted dataset containing sentence text and entity annotations.

### Key Steps

- Tokens are aligned to character-level entity spans.
- Predictions are mapped back to human-readable NER tags.
- `classification_report` from `sklearn` gives detailed precision, recall, and F1-scores.

### 📈 Sample Output

```
              precision    recall  f1-score   support

       B-LOC     1.0000    0.5000    0.6667         2
       B-ORG     0.5000    0.8571    0.6316         7
       B-PER     0.6000    0.6000    0.6000         5
       I-LOC     0.0000    0.0000    0.0000         1
       I-PER     0.6667    0.3333    0.4444         6
           O     0.9524    0.9524    0.9524        21
```

---

## 🧪 Example Inference

Prints actual tokens along with their predicted and ground truth labels for manual inspection.

```python
print("Tokens:", input_tokens)
print("Predicted:", pred_labels.tolist())
print("Actual:   ", true_labels.tolist())
```

---

## 🧰 Dependencies

Make sure to install the following Python packages:

```bash
pip install torch transformers pandas scikit-learn
```

---

## 📌 Key Concepts Demonstrated

- Token classification using Hugging Face Transformers
- Handling subword alignment for token-level NER tasks
- Evaluation using offset mappings and entity spans
- Custom PyTorch dataset and dataloader implementation

---

## 📎 License

MIT License © 2025

---

For questions or issues, please open an [issue](https://github.com/your-repo/issues) or contact the author.
