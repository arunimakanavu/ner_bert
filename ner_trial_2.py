from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import torch
from sklearn.metrics import classification_report

# Load dataset
with open('./ner/ner_eval.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load model and tokenizer
model_path = "./ner/ner_transformer_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none")  # no aggregation

# Label mapping
label_map = {
    "LABEL_0": "B-PER",
    "LABEL_1": "I-PER",
    "LABEL_2": "B-ORG",
    "LABEL_3": "I-ORG",
    "LABEL_4": "O",
    "LABEL_5": "B-LOC",
    "LABEL_6": "I-LOC"
}

true_labels = []
pred_labels = []

print("\n---- Token-level Evaluation ----")
for example in data:
    sentence = example['text']
    tokens = tokenizer(sentence, return_offsets_mapping=True, return_tensors='pt', truncation=True)
    offset_mapping = tokens.pop("offset_mapping")[0].tolist()
    input_ids = tokens['input_ids'][0]

    # Run model
    with torch.no_grad():
        output = model(**tokens).logits
    predictions = output.argmax(dim=2)[0]

    # Convert model predictions to labels
    pred_tag_seq = [model.config.id2label[pred.item()] for pred in predictions]

    # Assign gold labels by matching offsets
    char_to_label = {}
    for entity in example['entities']:
        ent_start = sentence.lower().find(entity['text'].lower())
        ent_end = ent_start + len(entity['text'])
        char_to_label[(ent_start, ent_end)] = label_map.get(entity['label'], entity['label'])

    # Align tokens to gold labels using offset
    for i, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue  # skip special tokens like [CLS], [SEP]

        gold_label = "O"
        for (ent_start, ent_end), label in char_to_label.items():
            if start >= ent_start and end <= ent_end:
                gold_label = label
                break

        true_labels.append(gold_label)
        pred_labels.append(label_map.get(pred_tag_seq[i], pred_tag_seq[i]))

# Report
print(classification_report(true_labels, pred_labels, digits=4))
