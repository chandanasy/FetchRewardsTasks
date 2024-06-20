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


**Sample Output:**
The script will generate embeddings for the input sentences and save them in a specified format (e.g., `.npy` file). These embeddings can be used for various downstream tasks such as clustering, classification, etc.

### Task 2

The Task 2 directory contains code for multi-task learning expansion task.
**To run the code for Task 2:**
    ```bash
    python Task2/main.py
    ```

**Description:**
The `main.py` script orchestrates the entire process of loading the dataset, training the model, and evaluating its performance. 

### Task 3

Task 3 involves training considerations for the IMDb multi-task model. The writeup is provided for task 3. No separate script is provided for this task.

### Task 4

The Task 2 directory also contains a script `main_t4.py` for running the model with layer-wise learning rates.

**To run the code with layer-wise learning rates for Task 4:**
    ```bash
    python Task2/main_t4.py
    ```

**Description:**
The `main_t4.py` script is similar to `main.py` but incorporates layer-wise learning rates for fine-tuning the model. It includes:


Feel free to reach out if you have any questions or issues running the code!



