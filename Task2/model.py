import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer

class MultiTaskModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes_task_a=4, num_classes_task_b=2):
        super(MultiTaskModel, self).__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.classifier_task_a = nn.Linear(self.encoder.config.hidden_size, num_classes_task_a) #4 class classification
        self.classifier_task_b = nn.Linear(self.encoder.config.hidden_size, num_classes_task_b) #2 class sentimental analysis
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  #use output of CLS token
        logits_task_a = self.classifier_task_a(pooled_output)
        logits_task_b = self.classifier_task_b(pooled_output)
        return logits_task_a, logits_task_b
