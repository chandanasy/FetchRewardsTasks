import torch
from transformers import BertModel, BertTokenizer

#load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

#sample sentences to encode
sentences = [
    "I'm hardworking and I know my algorithms really well.",
    "I always get the job done.",
    "I think you should really hire me.",
    "It's a decision you will not regret."
]

#tokenize sentences
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)

#forward pass -> get hidden states
with torch.no_grad():
    outputs = model(**inputs)

#get embeddings of CLS token i.e index 0 from last hidden state
embeddings = outputs.last_hidden_state[:, 0, :]

#display embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}\n")

