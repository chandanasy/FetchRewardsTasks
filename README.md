Sure, here's the completed `README.md` with detailed instructions for Task 1, including running the code and what it does.

```markdown
# FetchRewardsTasks

Tasks for the Apprenticeship program at FETCH.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/chandanasy/FetchRewardsTasks.git
    cd FetchRewardsTasks
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

### Task 1

The Task 1 directory contains the script `SentEmbed.py`, which is used for generating sentence embeddings.

**To run the code for Task 1:**
```bash
python Task1/SentEmbed.py
```

**Description:**
The `SentEmbed.py` script is designed to process text data and generate embeddings for sentences using a pre-trained model. It performs the following steps:

1. **Imports necessary libraries and models:**
   - Imports libraries such as `torch`, `transformers`, etc.
   - Loads the pre-trained model and tokenizer.

2. **Loads and preprocesses the data:**
   - Loads the dataset.
   - Tokenizes the sentences to prepare them for embedding generation.

3. **Generates embeddings:**
   - Passes the tokenized sentences through the pre-trained model to generate embeddings.
   - Saves the embeddings to a file for further use or analysis.

**Sample Output:**
The script will generate embeddings for the input sentences and save them in a specified format (e.g., `.npy` file). These embeddings can be used for various downstream tasks such as clustering, classification, etc.

### Task 2

The Task 2 directory contains multiple scripts for data preparation, model training, and evaluation.

**To run the code for Task 2:**
```bash
python Task2/main.py
```

**Description:**
The `main.py` script orchestrates the entire process of loading the dataset, training the model, and evaluating its performance. It includes the following steps:

1. **Data Preparation:**
   - Loads and preprocesses the IMDb dataset.
   - Tokenizes the text and splits it into training and validation sets.

2. **Model Training:**
   - Defines a multi-task learning model with separate heads for sentence classification and sentiment analysis.
   - Trains the model on the preprocessed data.

3. **Evaluation:**
   - Evaluates the trained model on the validation set.
   - Prints the performance metrics for both tasks.

### Task 3

Task 3 involves training considerations for the IMDb multi-task model. The training considerations are explained in the code comments. No separate script is provided for this task.

### Task 4

The Task 2 directory also contains a script `main_t4.py` for running the model with layer-wise learning rates.

**To run the code with layer-wise learning rates for Task 4:**
```bash
python Task2/main_t4.py
```

**Description:**
The `main_t4.py` script is similar to `main.py` but incorporates layer-wise learning rates for fine-tuning the model. It includes:

1. **Layer-wise Learning Rate Setup:**
   - Sets different learning rates for the transformer encoder and task-specific heads.

2. **Model Training:**
   - Trains the model with the specified layer-wise learning rates.
   - Saves checkpoints during training.

3. **Evaluation:**
   - Evaluates the model on the validation set.
   - Prints the performance metrics for both tasks.

Feel free to reach out if you have any questions or issues running the code!
```

Place this `README.md` file in the root directory of your GitHub repository. This will help users understand how to set up their environment and run your code effectively.
To run the code for Task 1, execute the following command:
```bash
python Task1/SentEmbed.py


