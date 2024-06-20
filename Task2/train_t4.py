import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os

def get_optimizer_with_layerwise_lr(model, base_lr=2e-5, encoder_lr=2e-5, classifier_lr=1e-4):
    no_decay = ["bias", "LayerNorm.weight"]
    #different learning rates for different layers
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": encoder_lr,
        },
        {
            "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": encoder_lr,
        },
        {
            "params": model.classifier_task_a.parameters(),
            "weight_decay": 0.01,
            "lr": classifier_lr,
        },
        {
            "params": model.classifier_task_b.parameters(),
            "weight_decay": 0.01,
            "lr": classifier_lr,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr)
    return optimizer

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def train_with_checkpoints(model, dataloader, optimizer, num_epochs=5, save_interval=200, checkpoint_dir="checkpoints", accumulation_steps=4):
    loss_fn = nn.CrossEntropyLoss()  #setting loss function
    model.train()
    step = 0  #initialize step counter

    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # Creating learning rate scheduler with warmup

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            #get inputs and labels from batch
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels_a = batch['labels_a']
            labels_b = batch['labels_b']
            
            #forward pass
            logits_task_a, logits_task_b = model(input_ids, attention_mask)

            #compute loss
            loss_a = loss_fn(logits_task_a, labels_a)
            loss_b = loss_fn(logits_task_b, labels_b)
            loss = loss_a + loss_b

            #backward pass
            loss.backward()
            #update weights and gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % save_interval == 0:
                save_checkpoint(model, optimizer, epoch, step, checkpoint_dir)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} completed")
