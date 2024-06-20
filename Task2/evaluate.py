import torch
from torch import nn

def evaluate_model(model, dataloader):
    model.eval()#set model to evaluation mode
    loss_fn = nn.CrossEntropyLoss()#define loss function
    total_loss_a = 0#set initial total loss for task a and b to 0
    total_loss_b = 0
    correct_a = 0#set initial correct predictions for task a and bto 0
    correct_b = 0
    total = 0 #set initial total number of samples to 0

    with torch.no_grad():#disable gradient calculation
        for batch in dataloader:#iterate over batches
            input_ids = batch['input_ids']#get input ids from batch
            attention_mask = batch['attention_mask']#get attention mask from batch
            labels_a = batch['labels_a']#get labels for task a and b from batch
            labels_b = batch['labels_b']
            
            logits_task_a, logits_task_b = model(input_ids, attention_mask)#get model predictions for both tasks
            loss_a = loss_fn(logits_task_a, labels_a)#calculate loss for task a and b
            loss_b = loss_fn(logits_task_b, labels_b)
            
            total_loss_a += loss_a.item()#total loss calculation 
            total_loss_b += loss_b.item()
            
            _, predicted_a = torch.max(logits_task_a, 1)#get predicted labels for task a and b 
            _, predicted_b = torch.max(logits_task_b, 1)
            
            correct_a += (predicted_a == labels_a).sum().item()#calculate correct predictions for task a and b
            correct_b += (predicted_b == labels_b).sum().item()
            total += labels_a.size(0)# total number of samples

    avg_loss_a = total_loss_a / len(dataloader)#calculate avg loss for a & b
    avg_loss_b = total_loss_b / len(dataloader)
    accuracy_a = correct_a / total
    accuracy_b = correct_b / total
    print(f"Average loss Task A: {avg_loss_a:.4f}, Accuracy Task A: {accuracy_a:.4f}")
    print(f"Average loss Task B: {avg_loss_b:.4f}, Accuracy Task B: {accuracy_b:.4f}")
