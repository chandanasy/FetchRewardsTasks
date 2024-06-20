import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
from collections import Counter

nltk.download('punkt')

def load_and_preprocess_data(subset_size=500):
    #using the IMdb dataset
    imdb = load_dataset('imdb')

    # Use the full dataset for balancing first
    texts = imdb['train']['text']
    sentiment_labels = imdb['train']['label']

    #split each entry into sentences since the task descriptions requires SENTENCES and not blocks of text
    sentences = []
    sentiment_sentences_labels = []

    for text, label in zip(texts, sentiment_labels):
        for sentence in sent_tokenize(text):
            sentences.append(sentence)
            sentiment_sentences_labels.append(label)

    #creating custom labels for task A based on keywords
    def create_custom_labels(sentences):
        labels = []
        for text in sentences:
            if "plot" in text.lower():
                labels.append(0)
            elif "acting" in text.lower():
                labels.append(1)
            elif "direction" in text.lower():
                labels.append(2)
            else:
                labels.append(3)  #other category if none of keywords are found
        return labels

    custom_labels = create_custom_labels(sentences)

    #encoding labels
    sentiment_label_encoder = LabelEncoder()
    custom_label_encoder = LabelEncoder()
    encoded_sentiment_labels = sentiment_label_encoder.fit_transform(sentiment_sentences_labels)
    encoded_custom_labels = custom_label_encoder.fit_transform(custom_labels)

    #during initiaL analysis it was found that the classes/labels were extremely unbalanced
    #therefore balancing classes for both tasks using oversampling
    def balance_classes(sentences, labels_a, labels_b):
        data = list(zip(sentences, labels_a, labels_b))
        df = pd.DataFrame(data, columns=['sentence', 'label_a', 'label_b'])
        max_size_a = df['label_a'].value_counts().max()
        max_size_b = df['label_b'].value_counts().max()
        max_size = max(max_size_a, max_size_b)
        
        lst = [df]
        for class_index, group in df.groupby('label_a'):
            lst.append(group.sample(max_size - len(group), replace=True))
        for class_index, group in df.groupby('label_b'):
            lst.append(group.sample(max_size - len(group), replace=True))
        df_new = pd.concat(lst)

        balanced_sentences = df_new['sentence'].tolist()
        balanced_labels_a = df_new['label_a'].tolist()
        balanced_labels_b = df_new['label_b'].tolist()
        
        return balanced_sentences, balanced_labels_a, balanced_labels_b

    balanced_sentences, balanced_custom_labels, balanced_sentiment_labels = balance_classes(sentences, custom_labels, sentiment_sentences_labels)

    #taking random subset of balanced dataset without exact splits
    balanced_subset_indices = random.sample(range(len(balanced_sentences)), subset_size)
    balanced_subset_sentences = [balanced_sentences[i] for i in balanced_subset_indices]
    balanced_subset_custom_labels = [balanced_custom_labels[i] for i in balanced_subset_indices]
    balanced_subset_sentiment_labels = [balanced_sentiment_labels[i] for i in balanced_subset_indices]

    return balanced_subset_sentences, balanced_subset_custom_labels, balanced_subset_sentiment_labels, custom_label_encoder, sentiment_label_encoder
