import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os

#creating checkpoints 
def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def train_with_checkpoints(model, dataloader, optimizer, num_epochs=4, save_interval=200, checkpoint_dir="checkpoints", accumulation_steps=4):
    loss_fn = nn.CrossEntropyLoss() #setting loss function 
    model.train()
    step = 0 #initialise step counter 

    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)     #creating learning rate scheduler with warmup


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
