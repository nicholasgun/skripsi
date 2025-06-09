import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import jaccard # Although jaccard similarity calculation is different for embeddings
from transformers import DebertaTokenizer, DebertaModel
from torch import nn
import fasttext
import fasttext.util
import re
from typing import List, Dict, Any, Tuple
import urllib.request
import sys
import gzip
import shutil
from scipy.spatial.distance import jaccard # Although jaccard similarity calculation is different for embeddings

class DeBERTaClassifier(nn.Module):
    """
    A classifier model based on DeBERTa for multi-label classification.
    
    This model uses a pre-trained DeBERTa model as the encoder and adds a 
    classification head on top with sigmoid activation for multi-label output.
    
    Args:
        num_labels (int): Number of classes in the multi-label classification task.
    """
    def __init__(self, num_labels):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) # 768 is the hidden size for deberta-base
        # Freeze all parameters in DeBERTa
        for param in self.deberta.parameters():
            param.requires_grad = False
        # Unfreeze encoder parameters for fine-tuning
        # Note: DeBERTa has a different architecture than BERT/RoBERTa
        # We'll unfreeze the last 3 encoder layers
        for layer in self.deberta.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Unlike BERT, DeBERTa doesn't have a pooler, so we need to take the last hidden state
        # and either use the [CLS] token (first token) or do mean pooling
        # Here we'll use the [CLS] token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        # Return raw logits for BCEWithLogitsLoss (sigmoid will be applied in the loss function)
        return self.classifier(cls_output)

class DeBERTaIssueDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts # Ensure texts is a list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def prepare_data(df, text_column='all_text', min_label_freq=0, max_label_len=100):
    if text_column in df.columns:
        df = df[[text_column, 'labels']]
        df = df[~df[text_column].apply(lambda x: x.startswith('nan') if isinstance(x, str) else False)]
    else:
        raise ValueError(f"Text column '{text_column}' not found in the DataFrame")
    
    df = df.dropna()
    texts = df[text_column]
    labels = df['labels'].apply(lambda x: x if isinstance(x, list) else [])

    label_distribution = Counter([label for labels in labels for label in labels])
    frequent_labels = [label for label, count in label_distribution.items() if count >= min_label_freq]
    
    filtered_labels = labels.apply(lambda x: [label for label in x if label in frequent_labels])
    label_length = filtered_labels.apply(len)
    length_mask = (label_length > 0) & (label_length <= max_label_len)
    
    texts = texts[length_mask].reset_index(drop=True)
    filtered_labels = filtered_labels[length_mask].reset_index(drop=True)
    
    return texts, filtered_labels

def preprocess_text(text: str) -> str:
    """
    Preprocess text for FastText embedding.
    """
    if not isinstance(text, str): # Handle potential non-string inputs
        text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t]', ' ', text)
    # Remove special characters but keep letters, numbers and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_embedding(text: str, model: fasttext.FastText._FastText) -> np.ndarray:
    """
    Get the sentence embedding for a text using the trained FastText model.
    Uses get_sentence_vector which is suitable after supervised training.
    """
    processed_text = preprocess_text(text)
    # If text is empty after preprocessing, return zero vector
    if not processed_text:
        return np.zeros(model.get_dimension())
    return model.get_sentence_vector(processed_text)

def get_embeddings(texts: pd.Series, model: fasttext.FastText._FastText) -> np.ndarray:
    """
    Get embeddings for a list of texts using the loaded FastText model.
    Uses get_sentence_vector for each text.
    """
    embeddings = []
    # Note: fasttext.get_sentence_vector doesn't benefit from explicit batching like transformers.
    # We process text by text.
    for i in tqdm(range(0, len(texts)), desc="Generating FastText embeddings"):
        embedding = get_text_embedding(str(texts.iloc[i]), model)
        embeddings.append(embedding)

    return np.array(embeddings).astype(np.float32) # Ensure float32 for similarity calculations

def download_with_progress(url: str, output_path: str) -> None:
    """
    Download a file with a tqdm progress bar.
    """
    try:
        print(f"Downloading from {url}")
        # Get content length from URL
        u = urllib.request.urlopen(url)
        total_size = int(u.info().get('Content-Length', 0))

        # Set up the tqdm progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading FastText Model") as pbar:
            def report_hook(count, block_size, total_size):
                pbar.update(block_size)

            # Download the file
            urllib.request.urlretrieve(url, output_path, reporthook=report_hook)
    except Exception as e:
        print(f"Error during download: {e}")
        # Try direct download without progress in case of error
        print("Attempting direct download...")
        urllib.request.urlretrieve(url, output_path)

def ensure_pretrained_vectors_downloaded(vector_file_name: str = 'cc.en.300.vec') -> str:
    """
    Ensures the FastText pre-trained vectors (.vec file) are downloaded and returns its path.
    Used as input for supervised training.
    """
    home_dir = os.path.expanduser("~")
    fasttext_dir = os.path.join(home_dir, ".fasttext")
    local_vector_path = os.path.join(os.getcwd(), vector_file_name)
    default_vector_path = os.path.join(fasttext_dir, vector_file_name)

    # Check various locations for the .vec file
    if os.path.exists(local_vector_path):
        print(f"Found pre-trained FastText vectors at {local_vector_path}")
        return local_vector_path
    elif os.path.exists(default_vector_path):
        print(f"Found pre-trained FastText vectors at {default_vector_path}")
        return default_vector_path
    else:
        print(f"Pre-trained FastText vectors ({vector_file_name}) not found, downloading...")

        # Ensure the FastText directory exists
        os.makedirs(fasttext_dir, exist_ok=True)

        # FastText download URL for English vectors (.vec.gz)
        gz_file_name = f"{vector_file_name}.gz"
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}"
        gz_path = os.path.join(fasttext_dir, gz_file_name)

        # Download with progress bar (use existing function)
        download_with_progress(url, gz_path)

        # Extract the gz file
        print(f"Extracting vector file to {default_vector_path}...")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(default_vector_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"Error during extraction: {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path) # Clean up downloaded archive on extraction error
            sys.exit(1)

        # Remove the gz file
        os.remove(gz_path)
        print(f"Vectors extracted to {default_vector_path}")

        # Try to create a symlink in current directory for convenience
        try:
            if not os.path.exists(local_vector_path):
                os.symlink(default_vector_path, local_vector_path)
                print(f"Created symlink to vectors in current directory")
        except (OSError, AttributeError) as e:
            print(f"Note: Could not create symlink in current directory: {e}")

        return default_vector_path

def ensure_pretrained_model_downloaded(model_name: str = 'cc.en.300.bin') -> str:
    """
    Ensures the FastText pre-trained model (.bin file) is downloaded and returns its path.
    Uses tqdm for download progress. This is used when no training is performed and no model path is given.
    """
    home_dir = os.path.expanduser("~")
    fasttext_dir = os.path.join(home_dir, ".fasttext")
    local_model_path = os.path.join(os.getcwd(), model_name)
    default_model_path = os.path.join(fasttext_dir, model_name)

    # Check various locations for the .bin file
    if os.path.exists(local_model_path):
        print(f"Found pre-trained FastText model at {local_model_path}")
        return local_model_path
    elif os.path.exists(default_model_path):
        print(f"Found pre-trained FastText model at {default_model_path}")
        return default_model_path
    else:
        print(f"Pre-trained FastText model ({model_name}) not found, downloading...")

        # Ensure the FastText directory exists
        os.makedirs(fasttext_dir, exist_ok=True)

        # FastText download URL for English model (.bin.gz)
        # Assuming 'cc.en.300.bin.gz' for the 'en' model.
        gz_file_name = f"{model_name}.gz"
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}"
        gz_path = os.path.join(fasttext_dir, gz_file_name)

        # Download with progress bar
        try:
            download_with_progress(url, gz_path)
        except Exception as e:
            print(f"Error during download with progress: {e}")
            print("Could not download the pre-trained model.")
            sys.exit(1) # Exit if download fails

        # Extract the gz file
        print(f"Extracting model file to {default_model_path}...")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(default_model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"Error during extraction: {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path) # Clean up downloaded archive on extraction error
            sys.exit(1)

        # Remove the gz file
        os.remove(gz_path)
        print(f"Model extracted to {default_model_path}")

        # Try to create a symlink in current directory for convenience
        try:
            if not os.path.exists(local_model_path):
                os.symlink(default_model_path, local_model_path)
                print(f"Created symlink to model in current directory")
        except (OSError, AttributeError) as e:
            print(f"Note: Could not create symlink in current directory: {e}")

        return default_model_path

def format_fasttext_data(texts: pd.Series, labels: List[List[str]], output_path: str):
    """
    Formats data for FastText supervised training and saves to a file.
    Format: __label__label1 __label__label2 ... text
    """
    print(f"Formatting data for FastText training, saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, label_list in tqdm(zip(texts, labels), total=len(texts), desc="Formatting data"):
            # Preprocess text
            processed_text = preprocess_text(str(text))
            # Format labels
            label_str = " ".join([f"__label__{l}" for l in label_list])
            # Write line
            f.write(f"{label_str} {processed_text}\\n")
    print("Data formatting complete.")

def train_fasttext_model(
    train_data_path: str,
    pretrained_vectors_path: str, # .vec file
    output_model_path: str, # .bin file
    epochs: int = 5,
    lr: float = 0.1,
    dim: int = 300,
    word_ngrams: int = 1,
    loss: str = 'softmax'
) -> str:
    """
    Train a FastText supervised model.

    Args:
        train_data_path: Path to the formatted training data.
        pretrained_vectors_path: Path to the pre-trained vectors file (e.g., cc.en.300.vec).
        output_model_path: Path where the trained model (.bin) will be saved.
        epochs: Number of training epochs.
        lr: Learning rate.
        dim: Dimension of vectors (must match pretrained vectors).
        word_ngrams: Max length of word n-grams.
        loss: Loss function ('softmax', 'ns', 'hs').

    Returns:
        Path to the trained model file (.bin).
    """
    print("\nStarting FastText supervised training...")
    print(f"  Train data: {train_data_path}")
    print(f"  Pretrained vectors: {pretrained_vectors_path}")
    print(f"  Output model: {output_model_path}")
    print(f"  Epochs: {epochs}, LR: {lr}, Dim: {dim}, Ngrams: {word_ngrams}, Loss: {loss}")

    # Check if dimension matches pretrained vectors
    # We'll load the vectors temporarily just to check the dimension if needed.
    # This adds overhead but prevents cryptic FastText errors.
    try:
        # Load only header to check dimension efficiently
        with open(pretrained_vectors_path, 'r', encoding='utf-8', errors='ignore') as f:
            num_vectors, vec_dim = map(int, f.readline().split())
        if dim != vec_dim:
            print(f"Warning: Provided --embedding_dim ({dim}) does not match pretrained vectors dimension ({vec_dim}).")
            print(f"Adjusting training dimension to {vec_dim} to match vectors.")
            dim = vec_dim # Adjust dim to match vectors
    except Exception as e:
        print(f"Warning: Could not read dimension from pretrained vectors file ({pretrained_vectors_path}): {e}")
        print("Proceeding with specified --embedding_dim, but training might fail if it doesn't match.")

    model = fasttext.train_supervised(
        input=train_data_path,
        pretrainedVectors=pretrained_vectors_path,
        epoch=epochs,
        lr=lr,
        dim=dim, # Use potentially adjusted dimension
        wordNgrams=word_ngrams,
        loss=loss,
        thread=os.cpu_count() or 1 # Use available cores
    )

    # Save the trained model (binary file)
    model.save_model(output_model_path)
    print(f"Training complete. Model saved to {output_model_path}")

    return output_model_path

def calculate_similarities(test_embeddings, train_embeddings, similarity_metric='cosine'):
    """
    Calculate similarities between test and train embeddings using different metrics.
    
    Args:
        test_embeddings: numpy array of shape (n_test, embedding_dim)
        train_embeddings: numpy array of shape (n_train, embedding_dim)
        similarity_metric: one of 'cosine', 'euclidean', 'jaccard'
        
    Returns:
        similarities: numpy array of shape (n_test, n_train)
    """
    if similarity_metric == 'cosine':
        return cosine_similarity(test_embeddings, train_embeddings)
    
    elif similarity_metric == 'euclidean':
        # Convert distances to similarities (higher is more similar)
        distances = euclidean_distances(test_embeddings, train_embeddings)
        max_dist = np.max(distances)
        if max_dist == 0:
            return np.ones_like(distances)
        return 1 - (distances / max_dist)
    
    elif similarity_metric == 'jaccard':
        # Jaccard similarity for continuous embeddings is not standard.
        # The version from 2_a binarized based on median. We'll replicate that here.
        # Be aware this is an approximation for Jaccard on continuous data.
        # We'll use the median of each dimension as a threshold
        binary_test = test_embeddings > np.median(test_embeddings, axis=0)
        binary_train = train_embeddings > np.median(train_embeddings, axis=0)
        
        similarities = np.zeros((len(test_embeddings), len(train_embeddings)))
        
        for i in range(len(test_embeddings)):
            for j in range(len(train_embeddings)):
                intersection = np.logical_and(binary_test[i], binary_train[j]).sum()
                union = np.logical_or(binary_test[i], binary_train[j]).sum()
                similarities[i, j] = intersection / union if union > 0 else 0
                
        return similarities
    
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

def calculate_label_based_metrics(test_labels, train_labels, similar_indices, k_values=[1, 3, 5]):
    """
    Calculate precision@k, recall@k, F1@k and other metrics based on label matching.
    
    Args:
        test_labels: list of label lists
        train_labels: list of label lists
        similar_indices: list of arrays containing indices of similar items
        k_values: list of k values for metrics
    """
    metrics = {}
    for k in k_values:
        metrics.update({
            f'precision@{k}': [],
            f'recall@{k}': [],
            f'f1@{k}': []
        })
    
    metrics.update({
        'avg_label_overlap': [],
        'total_matches': 0,
        'total_test_samples': len(test_labels)
    })
    
    for i, test_label_set in enumerate(test_labels):
        test_labels_set = set(test_label_set)
        if not test_labels_set:  # Skip empty test label sets
            continue
            
        retrieved_indices = similar_indices[i]
        
        for k in k_values:
            # Consider only top-k results
            top_k_indices = retrieved_indices[:k]
            
            # Calculate true positives across all top-k recommendations
            retrieved_labels_set = set()
            for idx in top_k_indices:
                retrieved_labels_set.update(train_labels[idx])
            
            true_positives = len(test_labels_set.intersection(retrieved_labels_set))
            
            # Calculate metrics
            precision = true_positives / len(retrieved_labels_set) if retrieved_labels_set else 0
            recall = true_positives / len(test_labels_set) if test_labels_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'precision@{k}'].append(precision)
            metrics[f'recall@{k}'].append(recall)
            metrics[f'f1@{k}'].append(f1)
        
        # Calculate label overlap for all top-k results
        label_overlaps = []
        for j, idx in enumerate(retrieved_indices):
            train_labels_set = set(train_labels[idx])
            if test_labels_set & train_labels_set:  # If there's an intersection
                overlap = len(test_labels_set & train_labels_set) / len(test_labels_set | train_labels_set)
                label_overlaps.append(overlap)
                
                # Count if there's at least one match
                if j == 0:  # Only count once per test sample
                    metrics['total_matches'] += 1
        
        avg_overlap = np.mean(label_overlaps) if label_overlaps else 0
        metrics['avg_label_overlap'].append(avg_overlap)
    
    # Calculate averages safely, handling potential division by zero if no valid samples
    num_valid_samples = sum(1 for tl in test_labels if tl) # Count non-empty test label lists
    if num_valid_samples == 0:
        print("Warning: No test samples with labels found. Metrics will be zero.")
        for k in k_values:
            metrics[f'avg_precision@{k}'] = 0.0
            metrics[f'avg_recall@{k}'] = 0.0
            metrics[f'avg_f1@{k}'] = 0.0
        metrics['avg_label_overlap'] = 0.0
        metrics['match_rate'] = 0.0
    else:
        for k in k_values:
            metrics[f'avg_precision@{k}'] = np.mean(metrics[f'precision@{k}'])
            metrics[f'avg_recall@{k}'] = np.mean(metrics[f'recall@{k}'])
            metrics[f'avg_f1@{k}'] = np.mean(metrics[f'f1@{k}'])

        metrics['avg_label_overlap'] = np.mean(metrics['avg_label_overlap'])
        # Match rate calculation based on the total number of test samples (including empty ones)
        metrics['match_rate'] = metrics['total_matches'] / metrics['total_test_samples'] if metrics['total_test_samples'] > 0 else 0

    return metrics

def create_stratification_labels(labels_list, min_samples_per_label=2):
    """
    Create stratification labels that ensure each label has enough samples.
    Only considers labels that appear frequently enough for stratification.
    """
    # Count label occurrences
    label_counts = Counter([label for labels in labels_list for label in labels])
    
    # Keep only labels that appear frequently enough
    frequent_labels = {label for label, count in label_counts.items() if count >= min_samples_per_label}
    
    # Create binary indicators only for frequent labels
    stratification_indicators = []
    for labels in labels_list:
        # Create indicator only for frequent labels
        indicator = tuple(sorted(label for label in labels if label in frequent_labels))
        # If no frequent labels, use a special category
        if not indicator:
            indicator = ('rare_combination',)
        stratification_indicators.append(indicator)
    
    return stratification_indicators

def calculate_and_evaluate_similarity(test_embeddings, filtered_train_embeddings, test_labels, original_train_labels, original_index_map, k_values, run_dir):
    """
    Calculates similarities between test and filtered train embeddings,
    maps indices back to original, evaluates metrics, and saves plots.
    
    Args:
        test_embeddings: numpy array (n_test, dim)
        filtered_train_embeddings: numpy array (n_filtered_train, dim)
        test_labels: list of lists (ground truth labels for test set)
        original_train_labels: list of lists (ground truth labels for the *original* full training set)
        original_index_map: dict mapping filtered train index -> original train index
        k_values: list of k values for evaluation
        run_dir: directory to save plots and results

    Returns:
        tuple: (results_dict, all_similar_indices_mapped, all_similarity_scores_mapped)
            results_dict: Dictionary containing aggregated metrics for each similarity type.
            all_similar_indices_mapped: Dict mapping metric -> list of lists of *original* train indices.
            all_similarity_scores_mapped: Dict mapping metric -> list of lists of similarity scores.
    """
    similarity_metrics = ['cosine', 'euclidean', 'jaccard']
    results = {} # Store aggregated metrics per similarity type
    all_similar_indices_mapped = {} # Store top-k original indices per metric
    all_similarity_scores_mapped = {} # Store corresponding top-k scores per metric
    max_k = max(k_values)

    if filtered_train_embeddings.shape[0] == 0:
        print("Warning: No filtered training embeddings to compare against. Skipping similarity calculation and evaluation.")
        # Return empty results
        for metric in similarity_metrics:
            results[metric] = {f'avg_precision@{k}': 0 for k in k_values}
            results[metric].update({
                f'avg_recall@{k}': 0 for k in k_values
            })
            results[metric].update({
                f'avg_f1@{k}': 0 for k in k_values
            })
            results[metric]['match_rate'] = 0
            results[metric]['avg_label_overlap'] = 0
            all_similar_indices_mapped[metric] = [[] for _ in range(len(test_embeddings))]
            all_similarity_scores_mapped[metric] = [[] for _ in range(len(test_embeddings))]
        return results, all_similar_indices_mapped, all_similarity_scores_mapped

    for metric in similarity_metrics:
        print(f"\nCalculating {metric} similarities...")
        # Calculate full similarity matrix between test and the filtered training set
        similarities = calculate_similarities(
            test_embeddings, filtered_train_embeddings, metric
        )
        
        # Get top-k indices and scores *within the filtered set*
        # Argsort gives indices of the smallest values first, use [::-1] for descending order
        num_filtered_train = filtered_train_embeddings.shape[0]
        current_max_k = min(max_k, num_filtered_train)

        # Get indices of top k scores for each test sample
        top_k_indices_filtered = np.argsort(similarities, axis=1)[:, -current_max_k:][:, ::-1] 
        # Get the actual scores corresponding to these indices
        top_k_scores = np.array([similarities[i, top_k_indices_filtered[i]] for i in range(len(test_embeddings))])

        # Map filtered indices back to original training set indices
        similar_original_indices_list = []
        for filtered_indices_row in top_k_indices_filtered:
            original_indices_row = [original_index_map[filt_idx] for filt_idx in filtered_indices_row]
            similar_original_indices_list.append(original_indices_row)
            
        # Store mapped indices and scores for this metric
        all_similar_indices_mapped[metric] = similar_original_indices_list
        all_similarity_scores_mapped[metric] = top_k_scores.tolist() # Store as list

        # Evaluate using original labels
        print(f"Evaluating metrics for {metric} similarity...")
        metrics = calculate_label_based_metrics(
            test_labels, 
            original_train_labels, # Pass the original full set of train labels
            similar_original_indices_list, # Pass the list of lists of mapped original indices
            k_values
        )

        # Store aggregated metrics
        results[metric] = metrics

    # Create comparison plots (similar to before)
    print("\nGenerating comparison plots...")
    for k in k_values:
        plt.figure(figsize=(12, 8))
        metrics_to_plot = [f'avg_precision@{k}', f'avg_recall@{k}', f'avg_f1@{k}']
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        for i, metric in enumerate(similarity_metrics):
            values = [results[metric].get(m, 0) for m in metrics_to_plot] # Use .get for safety if metric calculation failed
            plt.bar(x + i*width, values, width, label=f'{metric.capitalize()} Similarity')

        plt.ylabel('Score')
        plt.title(f'Comparison of Similarity Metrics at k={k} (Filtered Train Set)')
        plt.xticks(x + width, [m.split('@')[0].replace("avg_", "").capitalize() for m in metrics_to_plot])
        plt.legend()
        plt.grid(True, axis='y')
        plot_path = os.path.join(run_dir, f'similarity_metrics_comparison_k{k}_filtered_train.png')
        plt.savefig(plot_path)
        print(f"Saved comparison plot to {plot_path}")
        plt.close()

    # Save comparison results dictionary
    comparison_results_path = os.path.join(run_dir, 'similarity_metrics_comparison_filtered_train.json')
    with open(comparison_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved comparison metrics to {comparison_results_path}")

    return results, all_similar_indices_mapped, all_similarity_scores_mapped

def predict_labels_with_deberta(texts, model, tokenizer, mlb_deberta, device, batch_size, max_length, threshold=0.5):
    """
    Predict multi-labels for given texts using a trained DeBERTa model.

    Args:
        texts (pd.Series): Series of input texts.
        model (nn.Module): Trained DeBERTaClassifier model.
        tokenizer (transformers.PreTrainedTokenizer): DeBERTa tokenizer.
        mlb_deberta (MultiLabelBinarizer): Fitted MultiLabelBinarizer corresponding to the DeBERTa model's labels.
        device: Device to perform inference on.
        batch_size (int): Batch size for prediction.
        max_length (int): Maximum token length for tokenizer.
        threshold (float): Threshold for converting sigmoid outputs to binary labels.

    Returns:
        list[list[str]]: List of predicted label lists for each input text.
    """
    model.eval() # Set model to evaluation mode
    dataset = DeBERTaIssueDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds_binary = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting labels with DeBERTa"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            # Apply sigmoid and threshold
            predictions_binary = (torch.sigmoid(outputs) >= threshold).cpu().numpy()
            all_preds_binary.append(predictions_binary)

    all_preds_binary = np.vstack(all_preds_binary)

    # Convert binary predictions back to label names
    predicted_labels = mlb_deberta.inverse_transform(all_preds_binary)
    # Convert tuples back to lists
    predicted_labels_list = [list(labels) for labels in predicted_labels]

    return predicted_labels_list

def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}_{args.text_column}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_json(args.data_path)
    
    if args.text_column not in df.columns:
        available_columns = [col for col in df.columns if col.startswith('all_text')]
        print(f"Text column '{args.text_column}' not found. Available text columns: {available_columns}")
        if len(available_columns) == 0:
            raise ValueError("No text columns found in the data")
        args.text_column = available_columns[0]
        print(f"Using '{args.text_column}' instead")
    
    texts, filtered_labels_all = prepare_data(
        df, 
        text_column=args.text_column,
        min_label_freq=args.min_label_freq, 
        max_label_len=args.max_label_len
    )
    
    # Create stratification indicators *before* split
    print("\nPreparing stratified split...")
    stratification_indicators = create_stratification_labels(filtered_labels_all)
    
    try:
        # Try stratified split and convert to lists
        train_indices, test_indices = train_test_split(
            range(len(texts)),
            test_size=0.1,
            random_state=42,
            stratify=stratification_indicators
        )
        
        # Use indices to split both texts and labels
        original_train_texts = texts.iloc[train_indices].reset_index(drop=True) # Keep original training data
        test_texts = texts.iloc[test_indices].reset_index(drop=True)
        original_train_labels = [filtered_labels_all[i] for i in train_indices] # Keep original training labels
        test_labels = [filtered_labels_all[i] for i in test_indices]
        
        print("Successfully performed stratified split")
    except ValueError as e:
        print(f"Warning: Could not perform stratified split ({str(e)})")
        print("Falling back to random split")
        
        # Random split with indices
        train_indices, test_indices = train_test_split(
            range(len(texts)),
            test_size=0.1,
            random_state=42
        )
        
        # Use indices to split both texts and labels
        original_train_texts = texts.iloc[train_indices].reset_index(drop=True) # Keep original training data
        test_texts = texts.iloc[test_indices].reset_index(drop=True)
        original_train_labels = [filtered_labels_all[i] for i in train_indices] # Keep original training labels
        test_labels = [filtered_labels_all[i] for i in test_indices]

    print(f"\nOriginal Training Samples: {len(original_train_texts)}")
    print(f"Test Samples: {len(test_texts)}")

    # --- DeBERTa Prediction FIRST ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\nLoading DeBERTa model and resources for prediction...")
    deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

    with open(args.deberta_label_encoder_path, 'r') as f:
        encoder_data = json.load(f)
    mlb_deberta = MultiLabelBinarizer()
    mlb_deberta.classes_ = np.array(encoder_data['classes'])
    num_deberta_labels = len(mlb_deberta.classes_)

    selected_deberta_labels = None
    if args.deberta_selected_labels_path:
        print(f"Loading selected labels from {args.deberta_selected_labels_path}")
        with open(args.deberta_selected_labels_path, 'r') as f:
            selected_data = json.load(f)
        selected_deberta_labels = selected_data['selected_labels']
        mlb_deberta.classes_ = np.array([lbl for lbl in selected_deberta_labels if lbl in mlb_deberta.classes_])
        num_deberta_labels = len(mlb_deberta.classes_)
        print(f"Using {num_deberta_labels} selected labels for DeBERTa prediction.")
    else:
        print(f"Using all {num_deberta_labels} labels from the DeBERTa encoder.")

    deberta_model = DeBERTaClassifier(num_labels=num_deberta_labels).to(device)
    deberta_model.load_state_dict(torch.load(args.deberta_model_path, map_location=device))
    deberta_model.eval() 

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DeBERTa prediction!")
        deberta_model = torch.nn.DataParallel(deberta_model)

    print("\nPredicting labels for the test set using DeBERTa...")
    predicted_test_labels = predict_labels_with_deberta(
        texts=test_texts, 
        model=deberta_model,
        tokenizer=deberta_tokenizer,
        mlb_deberta=mlb_deberta,
        device=device,
        batch_size=args.batch_size,
        max_length=512,
        threshold=args.deberta_threshold
    )
    print(f"Finished predicting labels for {len(predicted_test_labels)} test samples.")
    
    # Cleanup DeBERTa model to free memory if possible
    del deberta_model
    del deberta_tokenizer
    del mlb_deberta
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # --- End DeBERTa Prediction ---

    # --- Filter Training Data based on ALL Predicted Test Labels ---
    print("\nFiltering training data based on globally predicted test labels...")
    all_predicted_labels_set = set(label for labels in predicted_test_labels for label in labels)
    print(f"Total unique labels predicted across test set: {len(all_predicted_labels_set)}")

    filtered_train_indices_original = [] # Store original indices
    filtered_train_texts_list = []
    filtered_train_labels_list = []

    for i, (text, labels) in enumerate(zip(original_train_texts, original_train_labels)):
        if any(label in all_predicted_labels_set for label in labels):
            filtered_train_indices_original.append(i) # Store original index
            filtered_train_texts_list.append(text)
            filtered_train_labels_list.append(labels)

    # Create filtered Series/lists
    filtered_train_texts = pd.Series(filtered_train_texts_list)
    filtered_train_labels = filtered_train_labels_list # Already a list
    
    # Map from filtered index (0 to N-1) back to original index
    original_index_map = {filtered_idx: original_idx for filtered_idx, original_idx in enumerate(filtered_train_indices_original)}

    print(f"Filtered Training Samples: {len(filtered_train_texts)}")
    if len(filtered_train_texts) == 0:
        print("Warning: Filtering resulted in zero training samples. Similarity search will be skipped.")
        # Optionally exit or handle this case appropriately
        return {} 
    # --- End Training Data Filtering ---

    # --- Initialize and Prepare FastText Model ---
    print("Loading/Downloading FastText model...")
    fasttext_model_path = None
    
    if args.training_epochs > 0:
        # --- Fine-tune FastText Model ---
        print("\n--- Starting FastText Fine-Tuning ---")
        
        # 1. Ensure Pretrained Vectors are available (required for train_supervised input)
        print("Ensuring pre-trained FastText vectors (.vec) are available...")
        pretrained_vectors_path = ensure_pretrained_vectors_downloaded('cc.en.300.vec')
        
        # 2. Format Training Data
        train_data_file = os.path.join(run_dir, "train_fasttext_for_finetuning.txt")
        # Use the ORIGINAL training data BEFORE filtering
        format_fasttext_data(original_train_texts, original_train_labels, train_data_file) 
        
        # 3. Train the Model
        fasttext_model_path = os.path.join(run_dir, "finetuned_fasttext_model.bin")
        train_fasttext_model(
            train_data_path=train_data_file,
            pretrained_vectors_path=pretrained_vectors_path, # .vec file
            output_model_path=fasttext_model_path, # .bin file
            epochs=args.training_epochs,
            lr=args.learning_rate,
            dim=args.embedding_dim, # Dimension for the trained model
            word_ngrams=args.word_ngrams,
            loss=args.loss_function
        )
        print(f"Fine-tuning complete. Model saved to {fasttext_model_path}")
        # --- End FastText Fine-Tuning ---
        
    elif args.fasttext_model_path:
        # Use a pre-existing model path if provided AND no training is done
        if os.path.exists(args.fasttext_model_path):
            print(f"Loading pre-existing FastText model from: {args.fasttext_model_path}")
            fasttext_model_path = args.fasttext_model_path
        else:
            print(f"Warning: Specified FastText model path '{args.fasttext_model_path}' not found.")
            print("Falling back to downloading the standard English model.")
            fasttext_model_path = None # Reset to trigger download below

    if not fasttext_model_path:
        # Download standard pre-trained .bin model if not trained and no valid path provided
        print("Downloading standard pre-trained English model (cc.en.300.bin)...")
        try:
            # Use the function that downloads the .bin model
            fasttext_model_path = ensure_pretrained_model_downloaded('cc.en.300.bin') 
            print(f"Using downloaded standard model: {fasttext_model_path}")
        except Exception as e: 
            print(f"Error ensuring standard FastText model is available: {e}")
            print("Please check your internet connection or provide a valid model path using --fasttext_model_path.")
            sys.exit(1) 

    print(f"\nLoading final FastText model from {fasttext_model_path}...")
    try:
        model = fasttext.load_model(fasttext_model_path)
        loaded_model_dim = model.get_dimension()
        # Check consistency if a pre-trained model was used/downloaded
        if args.training_epochs <= 0 and args.embedding_dim != loaded_model_dim:
             print(f"Warning: Loaded model dimension ({loaded_model_dim}) differs from --embedding_dim argument ({args.embedding_dim}). Using loaded model dimension.")
             args.embedding_dim = loaded_model_dim # Use the actual dimension of the loaded model
        print(f"FastText model loaded. Dimension: {loaded_model_dim}")
    except ValueError as e:
         print(f"Error loading FastText model: {e}")
         print("Ensure the path points to a valid FastText .bin file.")
         sys.exit(1)
    # --- End FastText Model Prep ---

    # --- Generate Embeddings (Test and Filtered Train) ---
    print("Generating embeddings with loaded FastText model...")
    # Embed the original test texts
    test_embeddings = get_embeddings(test_texts, model)
    # Embed ONLY the filtered training texts
    filtered_train_embeddings = get_embeddings(filtered_train_texts, model)
    print(f"Generated {test_embeddings.shape[0]} test embeddings.")
    print(f"Generated {filtered_train_embeddings.shape[0]} filtered training embeddings.")
    # --- End Embedding Generation ---

    # --- Compare Similarity Metrics (using filtered embeddings) ---
    print("Calculating similarities and evaluating metrics...")
    k_values = [1, 3, 5, 10] 
    
    # Call the modified comparison function (needs implementation below)
    # Pass original_train_labels and the original_index_map
    similarity_comparison_results, all_similar_indices_mapped, all_similarity_scores_mapped = calculate_and_evaluate_similarity(
        test_embeddings, filtered_train_embeddings, 
        test_labels, original_train_labels, # Pass ORIGINAL train labels for evaluation
        original_index_map, # Pass the map
        k_values, run_dir
    )

    # --- Save Detailed Results ---
    all_similarity_details = {}
    similarity_metrics = ['cosine', 'euclidean', 'jaccard'] 

    for metric in similarity_metrics:
        print(f"Processing results for {metric} similarity...")
        # Retrieve pre-calculated *mapped* indices and scores
        # Note: calculate_and_evaluate_similarity should return indices mapped back to original
        similar_original_indices = all_similar_indices_mapped[metric] 
        similarity_scores = all_similarity_scores_mapped[metric] # These scores correspond to the mapped indices

        # Metrics are already calculated in calculate_and_evaluate_similarity
        label_metrics = similarity_comparison_results[metric]

        # Print metrics 
        print(f"{metric.capitalize()} Similarity Metrics (using filtered train set):")
        print(f"Match Rate (at least one match): {label_metrics.get('match_rate', 0.0):.4f}") # Use .get for safety
        for k in k_values:
            print(f"Average Precision@{k}: {label_metrics.get(f'avg_precision@{k}', 0.0):.4f}")
            print(f"Average Recall@{k}: {label_metrics.get(f'avg_recall@{k}', 0.0):.4f}")
            print(f"Average F1@{k}: {label_metrics.get(f'avg_f1@{k}', 0.0):.4f}")
        print(f"Average Label Overlap: {label_metrics.get('avg_label_overlap', 0.0):.4f}")

        # Generate and save detailed similarity results per test sample
        metric_similarity_results = []
        # Use args.top_k for the detailed results file length
        for i, (original_indices_for_test, scores_for_test) in enumerate(zip(similar_original_indices, similarity_scores)):
            test_sample = {
                'test_text': test_texts.iloc[i],
                'test_labels': test_labels[i],
                'predicted_test_labels': predicted_test_labels[i], # Add predicted labels for context
                'similar_requests': []
            }
            test_labels_set = set(test_labels[i])
            
            # Ensure we don't try to access more items than available
            num_retrieved = min(args.top_k, len(original_indices_for_test))

            for j in range(num_retrieved):
                original_idx = original_indices_for_test[j]
                score = scores_for_test[j]
                
                # Use original_idx to fetch from original training data
                train_text = original_train_texts.iloc[original_idx]
                train_label_list = original_train_labels[original_idx] 
                train_labels_set = set(train_label_list)
                
                matching_labels = list(test_labels_set & train_labels_set)
                similar_request = {
                    'rank': j + 1,
                    'original_train_index': int(original_idx), # Store original index
                    'text': train_text,
                    'labels': train_label_list,
                    'similarity_score': float(score),
                    'matching_labels': matching_labels,
                    'has_matching_label': len(matching_labels) > 0
                }
                test_sample['similar_requests'].append(similar_request)
            metric_similarity_results.append(test_sample)

        all_similarity_details[metric] = metric_similarity_results
        # Use a NpEncoder for saving JSON to handle potential numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(os.path.join(run_dir, f'{metric}_similarity_results_filtered_train.json'), 'w') as f:
            try:
                json.dump(metric_similarity_results, f, indent=4, cls=NpEncoder)
            except TypeError as e:
                print(f"Error saving JSON for {metric}: {e}. Check data types.")
                # Fallback or simplified saving if needed
                # json.dump([str(item) for item in metric_similarity_results], f, indent=4) 

    # Prepare combined results dictionary
    results = {
        'text_column': args.text_column,
        'filtering_info': {
            'total_unique_predicted_labels': len(all_predicted_labels_set),
            'original_training_samples': len(original_train_texts),
            'filtered_training_samples': len(filtered_train_texts),
        },
        'similarity_comparison': similarity_comparison_results, # Metrics calculated by calculate_and_evaluate_similarity
        'fasttext_info': {
            'model_path': fasttext_model_path,
            'model_dimension': loaded_model_dim, # Save actual loaded dimension
            'training_epochs': args.training_epochs,
            'learning_rate': args.learning_rate if args.training_epochs > 0 else None,
            'word_ngrams': args.word_ngrams if args.training_epochs > 0 else None,
            'loss_function': args.loss_function if args.training_epochs > 0 else None,
        },
        'deberta_filtering': {
            'model_path': args.deberta_model_path,
            'embedding_dim': args.embedding_dim,
            'threshold': args.deberta_threshold,
        },
    }

    with open(os.path.join(run_dir, 'all_metrics_results_filtered_train.json'), 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy(i) for i in obj]
            return obj
        results_serializable = convert_numpy(results)
        json.dump(results_serializable, f, indent=4)

    print(f"\nAnalysis completed! Results saved to {run_dir}")

    # Return relevant data if needed downstream (adjust as necessary)
    return {
        'model': model,
        'test_embeddings': test_embeddings,
        'filtered_train_embeddings': filtered_train_embeddings,
        'original_index_map': original_index_map,
        'all_similar_indices_mapped': all_similar_indices_mapped,
        'all_similarity_scores_mapped': all_similarity_scores_mapped,
        'similarity_details': all_similarity_details,
        'results_dir': run_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare change requests using SBERT embeddings and similarity metrics, with optional DeBERTa label filtering.')
    
    parser.add_argument('--data_path', type=str, 
                        default="/kaggle/input/kubernetes-final-bug-data-without-comments/cleaned_data_with_changed_files_no_comments.json",
                        help='Path to the JSON data file')
    parser.add_argument('--text_column', type=str, default='all_text_0.5',
                        help='Column name with the text data to use for training')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    
    parser.add_argument('--min_label_freq', type=int, default=5,
                        help='Minimum frequency for a label to be considered')
    parser.add_argument('--max_label_len', type=int, default=5,
                        help='Maximum number of labels per sample')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DeBERTa prediction (FastText embedding is not batched)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of similar requests to find for each test sample')
    
    # FastText Model Loading (Optional)
    parser.add_argument('--fasttext_model_path', type=str, default=None,
                        help='Optional path to load a pre-existing FastText model (.bin file). Used only if --training_epochs is 0.')

    # DeBERTa Filtering Settings
    parser.add_argument('--deberta_model_path', type=str,
                        default="/kaggle/input/deberta-bug-with-fs-10-lables-max-length-2/pytorch/default/1/best_model_all_text_0.5.pt",
                        help='Path to the trained DeBERTa model state_dict (.pt file). Uses default if not specified.')
    parser.add_argument('--deberta_label_encoder_path', type=str,
                        default='/kaggle/input/deberta-bug-with-fs-10-lables-max-length-2/pytorch/default/1/label_encoder.json',
                        help='Path to the DeBERTa label encoder (.json file). Uses default if not specified.')
    parser.add_argument('--deberta_selected_labels_path', type=str,
                        default="/kaggle/input/deberta-bug-with-fs-10-lables-max-length-2/pytorch/default/2/selected_labels.json",
                        help='Optional path to the selected labels JSON file if feature selection was used for DeBERTa. Uses default if not specified.')
    parser.add_argument('--deberta_threshold', type=float, default=0.5,
                        help='Threshold for DeBERTa label prediction')

    # FastText Training Arguments (Copied from 2_a)
    parser.add_argument('--training_epochs', type=int, default=0, # Default to 0 (no training)
                        help='Number of epochs for FastText supervised training. Set > 0 to enable training.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for FastText training (used only if training_epochs > 0)')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of embeddings. If training, this defines output dim. If loading, it should match the loaded model.')
    parser.add_argument('--word_ngrams', type=int, default=2,
                        help='Max length of word n-grams for FastText training (used only if training_epochs > 0)')
    parser.add_argument('--loss_function', type=str, default='softmax', choices=['softmax', 'ns', 'hs'],
                        help='Loss function for FastText training (used only if training_epochs > 0)')

    args, unknown = parser.parse_known_args()

    # Validate arguments related to training/loading
    if args.training_epochs <= 0 and args.fasttext_model_path and not os.path.exists(args.fasttext_model_path):
         print(f"Warning: --training_epochs is not > 0, and the specified --fasttext_model_path '{args.fasttext_model_path}' does not exist. Will attempt to download the default model.")
         args.fasttext_model_path = None # Reset path so download is triggered

    main(args)
