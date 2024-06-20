Tasks for the Apprentinceship program at FETCH.

Summary of Task 4:
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
