Tasks for the Apprentinceship program at FETCH.

Summary of Task 4:
Rationale for Layer-wise Learning Rates:
Fine-tuning Pre-trained Layers:

Pre-trained transformer layers have already learned useful representations during the pre-training phase. A smaller learning rate (encoder_lr) helps in fine-tuning these layers gently, ensuring that the pre-trained knowledge is not lost abruptly.
Training New Layers:

The classifier heads (classifier_task_a and classifier_task_b) are new and randomly initialized. These layers require a higher learning rate (classifier_lr) to learn task-specific features quickly.
Regularization:

Weight decay is used to prevent overfitting. Different weight decay values can be set for parameters that do not include bias and LayerNorm weights to maintain stability during training.

Layer-wise Learning Rates:

Base Learning Rate: 2e-5
Encoder Learning Rate: 2e-5
Classifier Learning Rate: 1e-4
Reason: Different learning rates help preserve the pre-trained knowledge in the encoder while allowing the newly added classifier heads to learn quickly.
Optimizer:
AdamW Optimizer: Suitable for transformer models, providing adaptive learning rates and weight decay.
Sceduler:
Linear Scheduler with Warmup: Helps stabilize training initially by gradually increasing the learning rate.
Training Process:

Accumulate Gradients: Used to update weights less frequently, which helps when working with smaller batch sizes.
Checkpointing: Ensures training progress is saved periodically, allowing resumption from the last checkpoint if needed.
By using layer-wise learning rates, the model can achieve better convergence and performance, as it allows different parts of the model to learn at their own optimal rates.
