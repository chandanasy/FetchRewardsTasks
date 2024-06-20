import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from DataPrep import load_and_preprocess_data
from model import MultiTaskModel
from train_t4 import train_with_checkpoints, get_optimizer_with_layerwise_lr
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
    # Load and preprocess data
    sentences, labels_a, labels_b, custom_label_encoder, sentiment_label_encoder = load_and_preprocess_data(subset_size=500)

    # Tokenize sentences
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128)

    # Create dataset and dataloader
    dataset = CustomDataset(encodings, labels_a, labels_b)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Define the model
    model = MultiTaskModel(model_name='bert-base-uncased', num_classes_task_a=4, num_classes_task_b=2)

    # Set up the optimizer with layer-wise learning rates
    optimizer = get_optimizer_with_layerwise_lr(model)

    # Train the model
    train_with_checkpoints(model, dataloader, optimizer, num_epochs=5, save_interval=200, checkpoint_dir="checkpoints")

    # Evaluate the model
    evaluate_model(model, dataloader)

if __name__ == '__main__':
    main()
