import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AdamW
from DataPrep import load_and_preprocess_data
from model import MultiTaskModel
from train import train_with_checkpoints
from evaluate import evaluate_model

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_a, labels_b):
        self.encodings = encodings
        self.labels_a = labels_a
        self.labels_b = labels_b

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_a'] = torch.tensor(self.labels_a[idx])
        item['labels_b'] = torch.tensor(self.labels_b[idx])
        return item

    def __len__(self):
        return len(self.labels_a)

def main():
    #load and preprocess data
    sentences, labels_a, labels_b, custom_label_encoder, sentiment_label_encoder = load_and_preprocess_data(subset_size=500)

    #tokenize sentences
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128)

    #create dataset and dataloader
    dataset = CustomDataset(encodings, labels_a, labels_b)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    #define model
    model = MultiTaskModel(num_classes_task_a=4, num_classes_task_b=2)

    #set up optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    #train model
    train_with_checkpoints(model, dataloader, optimizer, num_epochs=5, save_interval=200, checkpoint_dir="checkpoints")

    #evaluate the model
    evaluate_model(model, dataloader)

if __name__ == '__main__':
    main()
