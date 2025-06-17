import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import pdist, squareform
from transformers import DebertaTokenizer, DebertaModel
from torch import nn

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, model, label_encoder=None, max_length=512):
        """
        Args:
            texts: pandas Series or list of texts
            labels: list of label lists
            model: SBERT model
            label_encoder: MultiLabelBinarizer instance (optional)
            max_length: maximum sequence length
        """
        # Convert texts to list if it's a pandas Series
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        
        # Initialize or use provided label encoder
        if label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            self.labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(labels)
        
        self.model = model
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]  # Now a binary numpy array
        
        # Access tokenizer correctly, handling DataParallel wrapper
        actual_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        tokenizer = actual_model._first_module().tokenizer

        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class FineTunedSBERT(SentenceTransformer):
    def __init__(self, model_name='all-mpnet-base-v2', num_labels=20):
        super().__init__(model_name)
        for param in self.parameters():
            param.requires_grad = False
        
        for layer in self._first_module().auto_model.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        for param in self._first_module().auto_model.pooler.parameters():
            param.requires_grad = True
            
        hidden_size = self._first_module().auto_model.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self._first_module().auto_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

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
    
    # Filter out samples that have no labels after frequency filtering
    # This ensures we only keep samples with at least one retained label
    empty_labels_mask = filtered_labels.apply(len) > 0
    
    # Apply the empty labels mask first
    filtered_labels = filtered_labels[empty_labels_mask].reset_index(drop=True)
    texts = texts[empty_labels_mask].reset_index(drop=True)
    
    # Then apply the label length filter
    label_length = filtered_labels.apply(len)
    length_mask = (label_length > 0) & (label_length <= max_label_len)
    
    texts = texts[length_mask].reset_index(drop=True)
    filtered_labels = filtered_labels[length_mask].reset_index(drop=True)
    
    return texts, filtered_labels

def get_embeddings(texts, model, batch_size=32):
    # Determine the actual model and device, handling DataParallel
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.eval() # Set the underlying model to evaluation mode
    device = next(actual_model.parameters()).device

    # Use the original model (potentially wrapped) for the forward pass if using multiple GPUs
    # DataParallel handles the distribution automatically
    inference_model = model if isinstance(model, torch.nn.DataParallel) else actual_model
    if isinstance(inference_model, torch.nn.DataParallel):
        print(f"Using {torch.cuda.device_count()} GPUs for embedding generation!")

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size].tolist()
            
            # Use tokenizer from the actual_model
            tokenizer = actual_model._first_module().tokenizer
    
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=actual_model.max_seq_length, # Use actual model's max_seq_length
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Directly get embeddings from the base transformer model's output
            # If using DataParallel (inference_model is wrapped), it will handle the multi-GPU forward pass
            # We access the underlying module's auto_model to get the correct output structure
            base_transformer = actual_model._first_module().auto_model
            outputs = base_transformer(input_ids=input_ids, attention_mask=attention_mask)

            pooled_output = outputs.pooler_output
            
            embeddings.append(pooled_output.cpu().numpy())
    
    return np.vstack(embeddings)

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
        # For Jaccard, we need to binarize the embeddings
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

def train_sbert_epoch(model, train_loader, criterion, optimizer, device, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if (i + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_loader)

def validate_sbert(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return total_loss / len(val_loader)

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
    
    # Calculate averages
    for k in k_values:
        metrics[f'avg_precision@{k}'] = np.mean(metrics[f'precision@{k}'])
        metrics[f'avg_recall@{k}'] = np.mean(metrics[f'recall@{k}'])
        metrics[f'avg_f1@{k}'] = np.mean(metrics[f'f1@{k}'])
    
    metrics['avg_label_overlap'] = np.mean(metrics['avg_label_overlap'])
    metrics['match_rate'] = metrics['total_matches'] / metrics['total_test_samples']
    
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

def predict_labels_with_deberta(texts, model, tokenizer, mlb_deberta, device, batch_size, max_length, threshold=0.5, test_labels=None, selected_labels_set=None):
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
        test_labels (list, optional): List of test labels if available, used for feature selection filtering.
        selected_labels_set (set, optional): Set of selected label names if feature selection is enabled.

    Returns:
        list[list[str]]: List of predicted label lists for each input text.
        pd.Series: Filtered text samples (if feature selection was applied)
        list: Filtered test labels (if feature selection was applied and test_labels was provided)
    """
    # If we have feature selection enabled and test labels available, filter samples
    filtered_texts = texts
    filtered_test_labels = test_labels
    filtered_indices = None
    
    if selected_labels_set and test_labels:
        print("\nApplying feature selection filtering to test samples before prediction...")
        # Only keep test samples that have at least one label in the selected labels set
        filtered_indices = []
        filtered_texts_list = []
        filtered_test_labels_list = []
        
        for i, (text, labels) in enumerate(zip(texts, test_labels)):
            # Check if any of the sample's labels are in the selected labels set
            matching_labels = [label for label in labels if label in selected_labels_set]
            if matching_labels:  # Keep only samples with at least one matching selected label
                filtered_indices.append(i)
                filtered_texts_list.append(text)
                filtered_test_labels_list.append(matching_labels)  # Keep only the matching labels
                
        # Convert to pandas Series for consistency
        if filtered_texts_list:
            filtered_texts = pd.Series(filtered_texts_list)
            filtered_test_labels = filtered_test_labels_list
            print(f"Kept {len(filtered_texts)} test samples out of {len(texts)} after feature selection filtering")
        else:
            print("Warning: No test samples have any labels matching the feature selection!")
            # Return empty results if no samples remain
            return [], filtered_texts, filtered_test_labels
    
    model.eval() # Set model to evaluation mode
    dataset = DeBERTaIssueDataset(filtered_texts, tokenizer, max_length)
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
    
    # Convert tuples to lists for consistency
    predicted_labels = [list(labels) for labels in predicted_labels]
    
    # If we filtered the test samples, but we need to return predictions in the original order,
    # we'd need to map the predictions back to their original positions. This is optional.
    
    return predicted_labels, filtered_texts, filtered_test_labels

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

    # Create a set of selected labels for faster lookup
    selected_labels_set = None
    if selected_deberta_labels:
        selected_labels_set = set(selected_deberta_labels)
        print(f"Creating selected labels set for feature selection filtering with {len(selected_labels_set)} labels")
    
    print("\nPredicting labels for the test set using DeBERTa...")
    predicted_test_labels, filtered_test_texts, filtered_test_labels = predict_labels_with_deberta(
        texts=test_texts, 
        model=deberta_model,
        tokenizer=deberta_tokenizer,
        mlb_deberta=mlb_deberta,
        device=device,
        batch_size=args.batch_size,
        max_length=512,
        threshold=args.deberta_threshold,
        test_labels=test_labels,
        selected_labels_set=selected_labels_set
    )
    
    # Update test_texts and test_labels if they were filtered during prediction
    if filtered_test_texts is not None and len(filtered_test_texts) != len(test_texts):
        test_texts = filtered_test_texts
        test_labels = filtered_test_labels
        print(f"Using filtered test set with {len(test_texts)} samples after feature selection")
    
    print(f"Finished predicting labels for {len(predicted_test_labels)} test samples.")
    
    # Cleanup DeBERTa model to free memory if possible
    del deberta_model
    del deberta_tokenizer
    del mlb_deberta
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # --- End DeBERTa Prediction ---

    # --- Apply Feature Selection from DeBERTa model (if available) ---
    # We already have selected_labels_set from earlier, so we'll reuse it
    if selected_labels_set:
        print("\nApplying feature selection from DeBERTa model to training data...")
        print(f"Number of selected labels from feature selection: {len(selected_labels_set)}")
        
        # Filter original training data to only include samples with at least one selected label
        fs_filtered_train_indices = []
        fs_filtered_train_texts = []
        fs_filtered_train_labels = []
        
        for i, (text, labels) in enumerate(zip(original_train_texts, original_train_labels)):
            # Check if any of the sample's labels are in the selected labels set
            matching_labels = [label for label in labels if label in selected_labels_set]
            if matching_labels:  # Keep only samples with at least one matching selected label
                fs_filtered_train_indices.append(i)
                fs_filtered_train_texts.append(text)
                fs_filtered_train_labels.append(matching_labels)  # Keep only the matching labels
        
        # Update the original training data to use the filtered set
        original_train_texts = pd.Series(fs_filtered_train_texts)
        original_train_labels = fs_filtered_train_labels
        
        # Calculate using filtered_test_texts if available, otherwise use test_texts
        original_test_size = len(filtered_test_texts) if 'filtered_test_texts' in locals() else len(test_texts)
        print(f"After feature selection filtering: {len(original_train_texts)} samples retained out of the original {len(texts) - original_test_size}")
        print(f"Removed {len(texts) - original_test_size - len(original_train_texts)} samples with no matching labels in feature selection")
        
        if len(original_train_texts) == 0:
            print("WARNING: No training samples retained after feature selection filtering. Cannot proceed.")
            return {}
    
    # --- Filter Training Data based on ALL Predicted Test Labels ---
    print("\nFiltering training data based on globally predicted test labels...")
    all_predicted_labels_set = set(label for labels in predicted_test_labels for label in labels)
    # If we have feature selection, intersect with selected labels
    if selected_labels_set:
        all_predicted_labels_set = all_predicted_labels_set.intersection(selected_labels_set)
        print(f"Total unique labels predicted across test set (filtered by feature selection): {len(all_predicted_labels_set)}")
    else:
        print(f"Total unique labels predicted across test set: {len(all_predicted_labels_set)}")

    filtered_train_indices_original = [] # Store original indices
    filtered_train_texts_list = []
    filtered_train_labels_list = []

    # Filter training samples to keep only those with at least one label matching the predicted test labels
    retained_labels_count = 0
    for i, (text, labels) in enumerate(zip(original_train_texts, original_train_labels)):
        # First filter the labels to keep only those in all_predicted_labels_set
        filtered_sample_labels = [label for label in labels if label in all_predicted_labels_set]
        
        # Only keep samples that have at least one label after filtering
        if len(filtered_sample_labels) > 0:
            filtered_train_indices_original.append(i) # Store original index
            filtered_train_texts_list.append(text)
            filtered_train_labels_list.append(filtered_sample_labels)  # Store only the filtered labels
            retained_labels_count += len(filtered_sample_labels)

    # Create filtered Series/lists
    filtered_train_texts = pd.Series(filtered_train_texts_list)
    filtered_train_labels = filtered_train_labels_list # Already a list
    
    print(f"After filtering, retained {retained_labels_count} label instances that match the predicted test labels")
    
    # Map from filtered index (0 to N-1) back to original index
    original_index_map = {filtered_idx: original_idx for filtered_idx, original_idx in enumerate(filtered_train_indices_original)}

    print(f"Filtered Training Samples: {len(filtered_train_texts)} (out of {len(original_train_texts)} samples after feature selection)")
    if len(filtered_train_texts) == 0:
        print("Warning: Filtering resulted in zero training samples. Similarity search will be skipped.")
        # Optionally exit or handle this case appropriately
        return {} 
    # --- End Training Data Filtering ---

    # --- Initialize and Prepare SBERT Model ---
    # Initialize label encoder *only* based on the filtered training labels now
    print("\nInitializing label encoder based on filtered training labels...")
    mlb_sbert = MultiLabelBinarizer()
    mlb_sbert.fit(filtered_train_labels) # Fit only on labels that remain
    print(f"Number of unique SBERT labels (post-filtering): {len(mlb_sbert.classes_)}")
    
    print("\nInitializing SBERT model...")
    model = FineTunedSBERT('all-mpnet-base-v2', num_labels=len(mlb_sbert.classes_)) # Use count from filtered labels
    model.max_seq_length = 512
    model.use_fast_tokenizer = True
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device already set

    sbert_fine_tuned = False
    if args.sbert_model_path:
        if os.path.exists(args.sbert_model_path):
            print(f"Loading pre-trained SBERT model from {args.sbert_model_path}...")
            try:
                # Adjust loading if the number of labels changed significantly due to filtering
                # This might require careful handling depending on how the model was saved
                # Option 1: Load weights except for the classifier head if sizes mismatch
                pretrained_dict = torch.load(args.sbert_model_path, map_location='cpu')
                model_dict = model.state_dict()
                
                # Filter out unnecessary keys and potentially size-mismatched classifier
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False) # Use strict=False to ignore non-matching keys/sizes
                print("Pre-trained SBERT model weights loaded (classifier may be reinitialized if size mismatched).")
            except Exception as e:
                 print(f"Error loading SBERT model: {e}. Check compatibility with filtered label set.")
                 print("Proceeding with base SBERT model.")
                 model = FineTunedSBERT('all-mpnet-base-v2', num_labels=len(mlb_sbert.classes_)) # Re-initialize if loading failed

            model.eval() 
            sbert_fine_tuned = True 
        else:
            print(f"Warning: SBERT model path {args.sbert_model_path} not found. Proceeding without loading.")
    
    model = model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for SBERT!")
        model = torch.nn.DataParallel(model)
    # --- End SBERT Model Prep ---

    # --- Optional SBERT Fine-tuning (on FILTERED data) ---
    best_loss = float('inf')
    train_losses = []
    test_losses = [] # Note: Test set isn't filtered, this validation might be less meaningful now
    if args.training_epochs > 0 and not args.sbert_model_path:
        # Create datasets using filtered training data and original test data
        # The SBERT MultiLabelDataset needs the SBERT model instance passed to it
        train_dataset = MultiLabelDataset(filtered_train_texts, filtered_train_labels, model, label_encoder=mlb_sbert)
        # For validation, use the original test set but with the SBERT label encoder
        # This evaluates how well the model generalizes to labels *after* being trained only on the filtered set
        test_dataset = MultiLabelDataset(test_texts, test_labels, model, label_encoder=mlb_sbert) 

        effective_batch_size = args.batch_size // 4 if args.batch_size >= 4 else 1
        gradient_accumulation_steps = 4 if args.batch_size >= 4 else 1
        
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, num_workers=0)
        
        criterion = BCEWithLogitsLoss()
        # Ensure optimizer uses parameters from the potentially wrapped model
        optimizer_params = model.module.parameters() if isinstance(model, torch.nn.DataParallel) else model.parameters()
        optimizer = AdamW(optimizer_params, lr=2e-5)
        
        print(f"\nStarting SBERT fine-tuning on FILTERED data for {args.training_epochs} epochs...")
        
        for epoch in range(args.training_epochs):
            print(f"\nEpoch {epoch+1}/{args.training_epochs}")
            train_loss = train_sbert_epoch(model, train_loader, criterion, optimizer, device, gradient_accumulation_steps)
            test_loss = validate_sbert(model, test_loader, criterion, device) # Validate on original test set
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            print(f"Filtered Train Loss: {train_loss:.4f}")
            print(f"Original Test Loss: {test_loss:.4f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), os.path.join(run_dir, 'best_sbert_model_filtered_train.pt'))
                print("Saved new best model (trained on filtered data)")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        sbert_fine_tuned = True 

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Filtered Train Loss')
        plt.plot(test_losses, label='Original Test Loss')
        plt.title('SBERT Training (Filtered Train) and Test Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = os.path.join(run_dir, 'sbert_filtered_training_losses.png')
        plt.savefig(plot_path)
        print(f"Saved SBERT training loss plot to {plot_path}")
        plt.close()
        
        best_model_path = os.path.join(run_dir, 'best_sbert_model_filtered_train.pt')
        if os.path.exists(best_model_path):
            print(f"Loading best SBERT model (trained on filtered data) from {best_model_path}")
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            base_model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print("Warning: Best model file not found after training. Using the final state.")
        model.eval() 
    elif args.sbert_model_path:
        print("Skipping SBERT fine-tuning as a pre-trained model was loaded.")
    else: 
        print("Skipping SBERT fine-tuning (training_epochs=0 and no pre-trained model path provided). Using base/loaded SBERT model.")
        model.eval()
    # --- End SBERT Fine-tuning ---

    # --- Generate Embeddings (Test and Filtered Train) ---
    print("\nGenerating embeddings with final SBERT model...")
    # Embed the original test texts
    test_embeddings = get_embeddings(test_texts, model, batch_size=args.batch_size)
    # Embed ONLY the filtered training texts
    filtered_train_embeddings = get_embeddings(filtered_train_texts, model, batch_size=args.batch_size)
    print(f"Generated {test_embeddings.shape[0]} test embeddings.")
    print(f"Generated {filtered_train_embeddings.shape[0]} filtered training embeddings.")
    # --- End Embedding Generation ---

    # --- Compare Similarity Metrics (using filtered embeddings) ---
    print("\nCalculating similarities and evaluating metrics...")
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
        print(f"\nProcessing results for {metric} similarity...")
        # Retrieve pre-calculated *mapped* indices and scores
        # Note: calculate_and_evaluate_similarity should return indices mapped back to original
        similar_original_indices = all_similar_indices_mapped[metric] 
        similarity_scores = all_similarity_scores_mapped[metric] # These scores correspond to the mapped indices

        # Metrics are already calculated in calculate_and_evaluate_similarity
        label_metrics = similarity_comparison_results[metric]

        # Print metrics 
        print(f"\n{metric.capitalize()} Similarity Metrics (using filtered train set):")
        print(f"Match Rate (at least one match): {label_metrics['match_rate']:.4f}")
        for k in k_values:
            print(f"Average Precision@{k}: {label_metrics[f'avg_precision@{k}']:.4f}")
            print(f"Average Recall@{k}: {label_metrics[f'avg_recall@{k}']:.4f}")
            print(f"Average F1@{k}: {label_metrics[f'avg_f1@{k}']:.4f}")
        print(f"Average Label Overlap: {label_metrics['avg_label_overlap']:.4f}")

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
        with open(os.path.join(run_dir, f'{metric}_similarity_results_filtered_train.json'), 'w') as f:
            json.dump(metric_similarity_results, f, indent=4)

    # Prepare combined results dictionary
    results = {
        'text_column': args.text_column,
        'filtering_info': {
            'total_unique_predicted_labels': len(all_predicted_labels_set),
            'selected_labels_from_fs': len(selected_labels_set) if selected_labels_set else 0,
            'original_test_samples': len(texts) - (len(texts) - len(test_texts)),
            'filtered_test_samples': len(filtered_test_texts if 'filtered_test_texts' in locals() else test_texts),
            'original_training_samples_before_fs': len(texts) - len(test_texts),
            'original_training_samples_after_fs': len(original_train_texts),
            'filtered_training_samples': len(filtered_train_texts),
        },
        'similarity_comparison': similarity_comparison_results, # Metrics calculated by calculate_and_evaluate_similarity
        'sbert_training_on_filtered': {
            'performed': args.training_epochs > 0 and not args.sbert_model_path,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_loss': float(best_loss) if args.training_epochs > 0 and 'best_loss' in locals() and best_loss != float('inf') else None
        },
        'deberta_info': {
            'model_path': args.deberta_model_path,
            'label_encoder_path': args.deberta_label_encoder_path,
            'selected_labels_path': args.deberta_selected_labels_path,
            'prediction_threshold': args.deberta_threshold,
            'num_predicted_labels_used': num_deberta_labels
        }
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
                        help='Batch size for generating embeddings and training/prediction')
    parser.add_argument('--training_epochs', type=int, default=0,
                        help='Number of epochs for SBERT fine-tuning (0 to skip if not loading model)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of similar requests to find for each test sample')
    
    # SBERT Model Loading (Optional)
    parser.add_argument('--sbert_model_path', type=str, default=None,
                        help='Optional path to load a pre-trained/fine-tuned SBERT model state_dict (.pt file). If provided, SBERT fine-tuning is skipped.')

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

    args, unknown = parser.parse_known_args()
    results = main(args)