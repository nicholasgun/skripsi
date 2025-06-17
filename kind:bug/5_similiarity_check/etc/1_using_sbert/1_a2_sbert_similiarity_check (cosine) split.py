import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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

class MultiLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-label classification with SBERT.
    Handles text inputs and their corresponding multi-label targets.
    """
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
        
        # Tokenize the text using the model's tokenizer
        # Handle model directly or through DataParallel wrapper
        if isinstance(self.model, torch.nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
            
        encoding = actual_model._first_module().tokenizer(
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
    """
    Extension of SentenceTransformer with a classification head for multi-label classification.
    Freezes most of the SBERT layers and only fine-tunes the last few transformer layers and pooler.
    """
    def __init__(self, model_name='all-mpnet-base-v2', num_labels=20):
        super().__init__(model_name)
        # Freeze all parameters initially
        for param in self.parameters():
            param.requires_grad = False
        
        # Handle case where self might be wrapped in DataParallel
        # This shouldn't happen during initialization, but adding check for completeness
        first_module = self._first_module()
        
        # Unfreeze only the last 3 transformer layers for fine-tuning
        for layer in first_module.auto_model.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Unfreeze the pooler layer
        for param in first_module.auto_model.pooler.parameters():
            param.requires_grad = True
            
        # Add a classification head on top of SBERT
        hidden_size = first_module.auto_model.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for classification
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            logits: Raw model outputs before sigmoid activation
        """
        # Handle case where self might be wrapped in DataParallel
        first_module = self._first_module()
        outputs = first_module.auto_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

def prepare_data(df, text_column='all_text', min_label_freq=0, max_label_len=100):
    """
    Prepares the data by filtering and processing text and labels.
    
    Args:
        df: Input DataFrame with text and labels columns
        text_column: Name of column containing text data
        min_label_freq: Minimum frequency threshold for labels to be included
        max_label_len: Maximum number of labels per sample
        
    Returns:
        filtered_df: DataFrame with filtered rows
        texts: Series of text samples
        filtered_labels: List of filtered label lists
    """
    # Handle None values by converting to defaults
    min_label_freq = 0 if min_label_freq is None else min_label_freq
    max_label_len = 100 if max_label_len is None else max_label_len
    
    # Extract relevant columns and remove 'nan' text entries
    if text_column in df.columns:
        df = df[[text_column, 'labels']]
        # Filter out entries that start with 'nan' string
        df = df[~df[text_column].apply(lambda x: x.startswith('nan') if isinstance(x, str) else False)]
    else:
        raise ValueError(f"Text column '{text_column}' not found in the DataFrame")
    
    # Verify we have valid data after filtering out 'nan' entries
    if len(df) == 0:
        print(f"WARNING: No valid data found after filtering entries with 'nan' text in column '{text_column}'")
        return df, pd.Series(), []
    
    # Remove rows with NaN values
    df = df.dropna()
    if len(df) == 0:
        print(f"WARNING: No valid data found after dropping rows with NaN values")
        return df, pd.Series(), []
        
    texts = df[text_column]
    
    # Ensure 'labels' column contains lists
    labels = df['labels'].apply(lambda x: x if isinstance(x, list) else ([x] if not pd.isna(x) else []))
    
    # Verify we have labels after conversion
    if all(len(x) == 0 for x in labels):
        print(f"WARNING: No valid labels found in the data. Check the format of the 'labels' column.")
        return df, texts, labels.tolist()

    # Count label occurrences to find frequent labels
    label_distribution = Counter([label for labels in labels for label in labels])
    frequent_labels = [label for label, count in label_distribution.items() if count >= min_label_freq]
    
    # Print label frequency information
    print(f"Found {len(frequent_labels)} labels with frequency >= {min_label_freq} out of {len(label_distribution)} total labels")
    
    # Filter out infrequent labels
    filtered_labels = labels.apply(lambda x: [label for label in x if label in frequent_labels])
    
    # Apply length-based filtering (remove rows with too many or zero labels)
    label_length = filtered_labels.apply(len)
    length_mask = (label_length > 0) & (label_length <= max_label_len)
    
    # Verify we have data after length filtering
    if length_mask.sum() == 0:
        print(f"WARNING: No samples left after filtering for label length (0 < length <= {max_label_len})")
        print(f"Label length distribution: {label_length.value_counts().sort_index()}")
        return df, pd.Series(), []
    
    # Apply the filtering and reset indices
    filtered_df = df[length_mask].reset_index(drop=True)
    texts = texts[length_mask].reset_index(drop=True)
    filtered_labels = filtered_labels[length_mask].reset_index(drop=True)
    
    return filtered_df, texts, filtered_labels

def get_embeddings(texts, model, batch_size=32):
    """
    Generate embeddings for a list of texts using the SBERT model.
    Processes texts in batches to avoid memory issues.
    
    Args:
        texts: List or Series of text samples
        model: SentenceTransformer model (or DataParallel wrapped model)
        batch_size: Number of samples to process at once
        
    Returns:
        embeddings_array: Numpy array of embeddings with shape (n_samples, embedding_dim)
    """
    # Handle DataParallel wrapper to get the base SentenceTransformer model
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
    
    # Set to evaluation mode
    model.eval()
    embeddings = []
    
    # Get the correct device
    device = next(model.parameters()).device
    
    # Handle empty input case
    if len(texts) == 0:
        print("WARNING: Empty text list provided for embedding generation")
        # Return an empty array with the correct embedding dimension
        embedding_dim = base_model._first_module().auto_model.config.hidden_size
        return np.zeros((0, embedding_dim))
    
    # Ensure batch_size is at least 1 and not larger than the dataset
    batch_size = max(1, min(batch_size, len(texts)))
    
    # Process texts in batches with no gradient computation
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            # Get current batch (handle Series or list input)
            if isinstance(texts, pd.Series):
                batch_texts = texts.iloc[i:i + batch_size].tolist()
            else:
                batch_texts = texts[i:i + batch_size]
            
            # Ensure all batch texts are strings
            batch_texts = [str(text) for text in batch_texts]
            
            # Tokenize the batch of texts using the base model's tokenizer
            encoding = base_model._first_module().tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move tensors to the appropriate device (CPU/GPU)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get embeddings - use model in a way that works for both DataParallel and regular models
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    # DataParallel doesn't have no_sync method, just use forward pass directly
                    outputs = model.module._first_module().auto_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                else:
                    outputs = base_model._first_module().auto_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                
            pooled_output = outputs.pooler_output
            
            # Store batch embeddings
            embeddings.append(pooled_output.cpu().numpy())
    
    # Concatenate all batch embeddings into a single array
    embeddings_array = np.vstack(embeddings)
    
    return embeddings_array

def find_similar_requests(test_embeddings, train_embeddings, train_labels, top_k=5):
    """
    Find similar requests using cosine similarity between embedding vectors.
    
    This function calculates the similarity between each test embedding and all training embeddings,
    then finds the top-k most similar training samples for each test sample.
    
    Args:
        test_embeddings: numpy array of shape (n_test, embedding_dim) containing test sample embeddings
        train_embeddings: numpy array of shape (n_train, embedding_dim) containing training sample embeddings
        train_labels: list of label lists for the training samples
        top_k: number of similar items to retrieve for each test sample
    
    Returns:
        similar_indices: list of arrays, where each array contains indices of top-k similar training samples
        similarity_scores: list of arrays, where each array contains similarity scores for the top-k samples
    """
    # Handle empty input case
    if len(test_embeddings) == 0 or len(train_embeddings) == 0:
        print("WARNING: Empty embeddings detected in similarity calculation")
        return [], []
    
    # Adjust top_k if it's larger than the available training samples
    adjusted_top_k = min(top_k, len(train_embeddings))
    if adjusted_top_k < top_k:
        print(f"WARNING: top_k={top_k} is greater than the number of training samples ({len(train_embeddings)})")
        print(f"Adjusting top_k to {adjusted_top_k}")
    
    # Calculate cosine similarity between all test and training embeddings
    # This returns a matrix of shape (n_test, n_train) where each row contains
    # similarity scores between a test sample and all training samples
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    similar_indices = []
    similarity_scores = []
    
    # For each test sample, find the top-k most similar training samples
    for i in range(len(test_embeddings)):
        # Get indices of top-k similar items, sorted by similarity (highest first)
        # np.argsort returns indices that would sort the array in ascending order,
        # so we take the last top_k elements and reverse them
        top_indices = np.argsort(similarities[i])[-adjusted_top_k:][::-1]
        similar_indices.append(top_indices)
        similarity_scores.append(similarities[i][top_indices])
    
    return similar_indices, similarity_scores

def train_sbert_epoch(model, train_loader, criterion, optimizer, device, gradient_accumulation_steps=4):
    """
    Train the SBERT model for one complete epoch.
    
    Args:
        model: The SBERT model to train
        train_loader: DataLoader containing training data
        criterion: Loss function (typically BCEWithLogitsLoss for multi-label classification)
        optimizer: Optimizer for parameter updates (typically AdamW)
        device: Device to run the training on (cuda or cpu)
        gradient_accumulation_steps: Number of batches to accumulate gradients before updating parameters
                                    (useful for simulating larger batch sizes with limited memory)
    
    Returns:
        Average loss over the entire epoch
    """
    # Set model to training mode (enables dropout, batch normalization updates, etc.)
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # Get total batch count for progress reporting
    total_batches = len(train_loader)
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training", total=total_batches)):
        # Move batch data to the appropriate device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss and normalize by gradient_accumulation_steps
        loss = criterion(outputs, labels)
        loss = loss / gradient_accumulation_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Update weights after accumulating gradients for specified number of steps
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Track total loss (multiply by gradient_accumulation_steps to get the actual batch loss)
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Free up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Handle remaining gradients if the dataset size is not divisible by gradient_accumulation_steps
    if (i + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Return average loss over all batches
    return total_loss / len(train_loader)

def validate_sbert(model, val_loader, criterion, device):
    """
    Evaluate the SBERT model on validation data.
    
    Args:
        model: The trained SBERT model to evaluate
        val_loader: DataLoader containing validation data
        criterion: Loss function (same as used in training)
        device: Device to run the validation on (cuda or cpu)
    
    Returns:
        Average validation loss over the entire validation set
    """
    # Set model to evaluation mode (disables dropout, uses running stats for batch norm, etc.)
    model.eval()
    total_loss = 0
    
    # Get total batch count for progress reporting
    total_batches = len(val_loader)
    
    # Disable gradient calculation for validation to save memory and computation
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", total=total_batches):
            # Move batch data to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Track total loss
            total_loss += loss.item()
            
            # Free up GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Return average loss over all validation batches
    return total_loss / len(val_loader)

def calculate_label_based_metrics(test_labels, train_labels, similar_indices, k_values=[1, 3, 5, 10]):
    """
    Calculate precision@k, recall@k, F1@k and other metrics based on label matching.
    
    This function evaluates how well the similarity search performs in terms of relevant label matching.
    For each test sample, it checks if the retrieved similar items have matching labels and
    calculates various evaluation metrics at different k values.
    
    Args:
        test_labels: list of label lists for test samples
        train_labels: list of label lists for training samples
        similar_indices: list of arrays containing indices of similar items for each test sample
        k_values: list of k values for precision@k, recall@k, and F1@k calculations
    
    Returns:
        Dictionary of metrics including precision@k, recall@k, F1@k, average overlap scores,
        and overall match rate
    """
    # Convert pandas Series to lists if needed
    if isinstance(test_labels, pd.Series):
        test_labels = test_labels.tolist()
    if isinstance(train_labels, pd.Series):
        train_labels = train_labels.tolist()
    
    # Handle empty input case
    is_test_labels_empty = len(test_labels) == 0
    is_train_labels_empty = len(train_labels) == 0
    is_similar_indices_empty = len(similar_indices) == 0 if similar_indices is not None else True
    
    if is_test_labels_empty or is_train_labels_empty or is_similar_indices_empty:
        print("WARNING: Empty input to calculate_label_based_metrics. Returning empty metrics.")
        empty_metrics = {
            'avg_label_overlap': 0.0,
            'total_matches': 0,
            'total_test_samples': 0,
            'match_rate': 0.0
        }
        for k in k_values:
            empty_metrics[f'avg_precision@{k}'] = 0.0
            empty_metrics[f'avg_recall@{k}'] = 0.0
            empty_metrics[f'avg_f1@{k}'] = 0.0
        return empty_metrics
    # Initialize dictionaries to store metrics for each k value
    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'recall@{k}': [] for k in k_values})
    metrics.update({f'f1@{k}': [] for k in k_values})
    metrics.update({
        'avg_label_overlap': [],
        'total_matches': 0,
        'total_test_samples': len(test_labels)
    })
    
    # Process each test sample
    for i, test_label_set in enumerate(test_labels):
        # Convert test labels to set for faster set operations
        test_labels_set = set(test_label_set)
        # Get indices of retrieved similar items for this test sample
        retrieved_indices = similar_indices[i]
        
        # Track matches at each k and collected relevant labels for recall calculation
        matches_at_k = [0] * len(k_values)
        recall_at_k = [set() for _ in k_values]
        label_overlaps = []
        
        # Examine each retrieved item
        for j, idx in enumerate(retrieved_indices):
            # Convert train labels to set for intersection operation
            train_labels_set = set(train_labels[idx])
            # Find matching labels (intersection of test and train label sets)
            matching_labels = test_labels_set & train_labels_set
            
            # If there are matching labels, update metrics
            if matching_labels:
                # Update metrics for each k value where this item is included
                for k_idx, k in enumerate(k_values):
                    if j < k:  # Only include this item if it's within the top-k
                        matches_at_k[k_idx] += 1  # Count as a relevant retrieval
                        recall_at_k[k_idx].update(matching_labels)  # Collect matched labels for recall
                
                # Calculate Jaccard similarity coefficient for label overlap
                overlap = len(matching_labels) / len(test_labels_set | train_labels_set)
                label_overlaps.append(overlap)
        
        # Calculate precision, recall, and F1 for each k value
        for k_idx, k in enumerate(k_values):
            # Precision@k - proportion of retrieved items that are relevant
            precision_at_k = matches_at_k[k_idx] / k if k > 0 else 0
            metrics[f'precision@{k}'].append(precision_at_k)
            
            # Recall@k - proportion of relevant labels that are retrieved
            recall = len(recall_at_k[k_idx]) / len(test_labels_set) if test_labels_set else 0
            metrics[f'recall@{k}'].append(recall)
            
            # F1@k - harmonic mean of precision and recall
            if precision_at_k + recall > 0:
                f1 = 2 * precision_at_k * recall / (precision_at_k + recall)
            else:
                f1 = 0
            metrics[f'f1@{k}'].append(f1)
        
        # Calculate average label overlap for this test sample
        avg_overlap = np.mean(label_overlaps) if label_overlaps else 0
        metrics['avg_label_overlap'].append(avg_overlap)
        
        # Count as a match if any similar item had matching labels
        if any(matches_at_k):
            metrics['total_matches'] += 1
    
    # Calculate average metrics across all test samples
    for k in k_values:
        metrics[f'avg_precision@{k}'] = np.mean(metrics[f'precision@{k}'])
        metrics[f'avg_recall@{k}'] = np.mean(metrics[f'recall@{k}'])
        metrics[f'avg_f1@{k}'] = np.mean(metrics[f'f1@{k}'])
    
    # Calculate overall average label overlap and match rate
    metrics['avg_label_overlap'] = np.mean(metrics['avg_label_overlap'])
    metrics['match_rate'] = metrics['total_matches'] / metrics['total_test_samples']
    
    return metrics

def plot_precision_at_k(metrics, k_values, run_dir):
    """
    Plot precision@k, recall@k, and F1@k values and save the visualization.
    
    Args:
        metrics: Dictionary containing the calculated evaluation metrics
        k_values: List of k values for which metrics were calculated
        run_dir: Directory path where the plot should be saved
    """
    # Create a new figure with appropriate size
    plt.figure(figsize=(12, 8))
    
    # Extract metrics for different k values
    avg_precisions = [metrics[f'avg_precision@{k}'] for k in k_values]
    avg_recalls = [metrics[f'avg_recall@{k}'] for k in k_values]
    avg_f1s = [metrics[f'avg_f1@{k}'] for k in k_values]
    
    # Plot each metric with different markers for better visibility
    plt.plot(k_values, avg_precisions, marker='o', label='Precision@k')
    plt.plot(k_values, avg_recalls, marker='s', label='Recall@k')
    plt.plot(k_values, avg_f1s, marker='^', label='F1@k')
    
    # Add title and labels
    plt.title('Evaluation Metrics at Different k Values')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # Display the plot (useful for interactive environments)
    plt.show()
    
    # Save the plot to disk for later reference
    plt.savefig(os.path.join(run_dir, 'metrics_at_k.png'))
    plt.close()

def create_stratification_labels(labels_list, min_samples_per_label=2):
    """
    Create stratification labels that ensure each label has enough samples for stratified splitting.
    Only considers labels that appear frequently enough for stratification.
    
    This function is useful for multi-label stratification when doing train/test splits
    to ensure that rare label combinations are properly distributed between splits.
    
    Args:
        labels_list: List of label lists, where each inner list contains labels for one sample
        min_samples_per_label: Minimum number of samples a label must appear in to be considered
                              for stratification
    
    Returns:
        List of tuple indicators for stratification, where each tuple represents the frequent labels
        for that sample. Samples with no frequent labels get a special 'rare_combination' indicator.
    """
    # Count how many times each label appears across all samples
    label_counts = Counter([label for labels in labels_list for label in labels])
    
    # Filter to only include labels that appear frequently enough
    frequent_labels = {label for label, count in label_counts.items() if count >= min_samples_per_label}
    
    # Create stratification indicators for each sample
    stratification_indicators = []
    for labels in labels_list:
        # Create a tuple indicator containing only the frequent labels for this sample
        indicator = tuple(sorted(label for label in labels if label in frequent_labels))
        
        # Handle samples with no frequent labels by assigning them to a special category
        if not indicator:
            indicator = ('rare_combination',)
            
        stratification_indicators.append(indicator)
    
    return stratification_indicators

def main(args):
    """
    Main execution function that orchestrates the entire SBERT similarity analysis workflow.
    
    This function performs the following steps:
    1. Setup environment and directories
    2. Load and prepare training and testing data
    3. Initialize and optionally fine-tune the SBERT model
    4. Generate embeddings for all samples
    5. Save embeddings to original dataframes if requested
    6. Find similar requests using cosine similarity
    7. Calculate and visualize metrics
    8. Save all results to disk
    
    Args:
        args: Command line arguments parsed by argparse
        
    Returns:
        Dictionary containing model, embeddings, results, and other artifacts
    """
    # Configure CUDA environment and set random seeds for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create result directories with timestamp to avoid overwriting
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}_{args.text_column}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {args.train_path}...")
    train_df = pd.read_csv(args.train_path)
    
    # Load testing data
    print(f"Loading testing data from {args.test_path}...")
    test_df = pd.read_csv(args.test_path)
    
    # Check if text column exists in both datasets
    # If the specified column doesn't exist, try to find a suitable alternative
    if args.text_column not in train_df.columns or args.text_column not in test_df.columns:
        # Look for columns that start with 'all_text' as potential alternatives
        available_train_columns = [col for col in train_df.columns if col.startswith('all_text')]
        available_test_columns = [col for col in test_df.columns if col.startswith('all_text')]
        # Find columns that exist in both datasets
        common_columns = list(set(available_train_columns) & set(available_test_columns))
        
        print(f"Text column '{args.text_column}' not found in both datasets.")
        if common_columns:
            # Use the first common column as a fallback
            args.text_column = common_columns[0]
            print(f"Using common column '{args.text_column}' instead")
        else:
            # If no suitable column is found, raise an error
            raise ValueError("No common text columns found in both datasets")
    
    # Prepare training data
    # Filter and process the training data according to specified parameters
    train_df, train_texts, train_labels = prepare_data(
        train_df, 
        text_column=args.text_column,
        min_label_freq=args.min_label_freq,  # Filter out infrequent labels in training
        max_label_len=args.max_label_len
    )
    
    # Check if we have any training data left after filtering
    if len(train_texts) == 0:
        raise ValueError(f"No training data left after filtering. Try lowering min_label_freq (currently {args.min_label_freq}) " 
                        f"or increasing max_label_len (currently {args.max_label_len}).")
    
    # Prepare testing data 
    # For test data, we don't filter by label frequency to preserve all test cases
    test_df, test_texts, test_labels = prepare_data(
        test_df, 
        text_column=args.text_column,
        min_label_freq=0,  # Don't filter test labels by frequency
        max_label_len=args.max_label_len
    )
    
    # Check if we have any test data left after filtering
    if len(test_texts) == 0:
        raise ValueError(f"No test data left after filtering. Check if max_label_len={args.max_label_len} is too restrictive.")
    
    # Print dataset sizes after filtering
    print(f"\nDataset sizes after filtering:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Testing samples: {len(test_texts)}")
    
    # Initialize label encoder with all labels first
    # MultiLabelBinarizer transforms label lists into binary indicator vectors
    print("\nInitializing label encoder...")
    mlb = MultiLabelBinarizer()
    
    # Check if we have any labels to fit
    if (isinstance(train_labels, pd.Series) and train_labels.empty) or (isinstance(test_labels, pd.Series) and test_labels.empty) or (len(train_labels) == 0 and len(test_labels) == 0):
        raise ValueError("No valid labels found in the datasets. Cannot proceed with model training.")
    
    # Fit on both train and test labels to ensure all possible labels are included
    train_labels_list = train_labels.tolist() if isinstance(train_labels, pd.Series) else train_labels
    test_labels_list = test_labels.tolist() if isinstance(test_labels, pd.Series) else test_labels
    mlb.fit(train_labels_list + test_labels_list)
    print(f"Number of unique labels: {len(mlb.classes_)}")
    if args.debug or len(mlb.classes_) < 20:  # Only print all labels if in debug mode or if there are few labels
        print("Labels:", mlb.classes_)
    else:
        print(f"First 10 labels: {mlb.classes_[:10]}")
    
    # Initialize SBERT model with correct number of labels
    # This creates a fine-tunable SBERT model with a classification head
    print("\nInitializing SBERT model...")
    model = FineTunedSBERT('all-mpnet-base-v2', num_labels=len(mlb.classes_))
    model.max_seq_length = 512  # Maximum sequence length for tokenization
    model.use_fast_tokenizer = True  # Use faster tokenization methods
    
    # Configure device(s) for training
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA device(s)")
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for parallel training")
            # Use DataParallel for multi-GPU training
            # Note: DataParallel distributes batches across GPUs and combines results
            # It's simpler than DistributedDataParallel but less efficient for large models
            device = torch.device('cuda')
            model = torch.nn.DataParallel(model)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("No CUDA devices available, using CPU")
    
    model = model.to(device)  # Move model to appropriate device
    
    # Create datasets with shared label encoder
    # This wrapper class handles tokenization and label encoding
    print("\nCreating datasets...")
    
    # We need to access the base model for tokenization (not the DataParallel wrapper)
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
        
    train_dataset = MultiLabelDataset(train_texts, train_labels, base_model, label_encoder=mlb)
    test_dataset = MultiLabelDataset(test_texts, test_labels, base_model, label_encoder=mlb)
    
    # Create data loaders for batch processing
    # Use gradient accumulation to simulate larger batch sizes with limited memory
    effective_batch_size = min(args.batch_size // 4, max(1, len(train_dataset) // 10))  # Ensure reasonable batch size
    gradient_accumulation_steps = 4  # Number of batches to accumulate before update
    
    print(f"\nUsing batch size of {effective_batch_size} with {gradient_accumulation_steps} gradient accumulation steps")
    print(f"Effective batch size: {effective_batch_size * gradient_accumulation_steps}")
    print(f"Using {args.num_workers} worker threads for data loading")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,  # Shuffle data during training for better convergence
        num_workers=args.num_workers,  # Number of parallel workers for data loading
        pin_memory=torch.cuda.is_available()  # Pin memory for faster data transfer to GPU
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Fine-tune SBERT if specified
    # Skip this section if training_epochs is 0
    if args.training_epochs > 0:
        # Configure training components
        criterion = BCEWithLogitsLoss()  # Binary cross-entropy loss for multi-label classification
        optimizer = AdamW(model.parameters(), lr=2e-5)  # AdamW optimizer with small learning rate
        
        # Track best model and loss history
        best_loss = float('inf')
        train_losses = []
        test_losses = []
        
        print(f"\nStarting SBERT fine-tuning for {args.training_epochs} epochs...")
        
        # Training loop for specified number of epochs
        for epoch in range(args.training_epochs):
            print(f"\nEpoch {epoch+1}/{args.training_epochs}")
            
            # Train for one epoch and get average training loss
            train_loss = train_sbert_epoch(
                model, train_loader, criterion, optimizer, device, gradient_accumulation_steps
            )
            
            # Evaluate on test data
            test_loss = validate_sbert(
                model, test_loader, criterion, device
            )
            
            # Store losses for plotting
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            
            # Save model if it's the best so far (based on test loss)
            if test_loss < best_loss:
                best_loss = test_loss
                # Handle DataParallel wrapped models - save the module's state_dict
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(run_dir, 'best_sbert_model.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(run_dir, 'best_sbert_model.pt'))
                print("Saved new best model")
            
            # Free up GPU memory after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot training curves to visualize model progress
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'training_curves.png'))
        plt.close()
        
        # Load best model for generating embeddings
        # This ensures we use the best model found during training, not just the final one
        if isinstance(model, torch.nn.DataParallel):
            # For DataParallel model, load to the wrapped module
            model.module.load_state_dict(torch.load(os.path.join(run_dir, 'best_sbert_model.pt')))
        else:
            model.load_state_dict(torch.load(os.path.join(run_dir, 'best_sbert_model.pt')))
    
    # Set model to evaluation mode for embedding generation
    model.eval()
    
    # Generate embeddings for training and testing sets
    print("\nGenerating embeddings with model...")
    # These will be used for similarity matching and saving
    # For very small datasets, adjust batch size to avoid empty batches
    embed_batch_size = min(args.batch_size, len(train_texts))
    print(f"Using embedding batch size of {embed_batch_size}")
    
    train_embeddings = get_embeddings(train_texts, model, batch_size=embed_batch_size)
    test_embeddings = get_embeddings(test_texts, model, batch_size=embed_batch_size)
    
    # Optionally create a subset of features if requested (for storage efficiency)
    if args.save_subset_features > 0 and args.save_subset_features < train_embeddings.shape[1]:
        print(f"\nReducing embeddings to {args.save_subset_features} features for storage efficiency...")
        # Just take the first N features for simplicity
        # Could be replaced with PCA or other dimensionality reduction method
        train_embeddings_subset = train_embeddings[:, :args.save_subset_features]
        test_embeddings_subset = test_embeddings[:, :args.save_subset_features]
        
        # Convert these to lists for storage in the dataframe (JSON-serializable)
        train_embeddings_list = [embedding.tolist() for embedding in train_embeddings_subset]
        test_embeddings_list = [embedding.tolist() for embedding in test_embeddings_subset]
        
        # Save the reduced embeddings as numpy arrays if requested
        if args.save_embeddings:
            np.save(os.path.join(run_dir, f'train_embeddings_dim{args.save_subset_features}.npy'), 
                   train_embeddings_subset)
            np.save(os.path.join(run_dir, f'test_embeddings_dim{args.save_subset_features}.npy'), 
                   test_embeddings_subset)
    else:
        # Convert full embeddings to list of lists for easier storage in dataframe
        train_embeddings_list = [embedding.tolist() for embedding in train_embeddings]
        test_embeddings_list = [embedding.tolist() for embedding in test_embeddings]
    
    # Load the original input CSV files to add embeddings to them
    print("\nAdding embeddings to the original input files...")
    original_train_df = pd.read_csv(args.train_path)
    original_test_df = pd.read_csv(args.test_path)
    
    # Create a dictionary mapping text to embedding for efficient lookup
    # This avoids having to recompute embeddings or do slow dataframe operations
    train_text_to_embedding = dict(zip(train_texts, train_embeddings_list))
    
    # Add embedding column to original training dataframe if the text column exists
    if args.text_column in original_train_df.columns:
        # Map each text in the original dataframe to its corresponding embedding
        original_train_df['embedding'] = original_train_df[args.text_column].map(
            lambda x: train_text_to_embedding.get(x, None))
        
        # Save the updated original train dataframe with embeddings
        train_embed_path = os.path.join(run_dir, os.path.basename(args.train_path).replace('.csv', '_with_embeddings.csv'))
        print(f"Saving training data with embeddings to {train_embed_path}")
        original_train_df.to_csv(train_embed_path, index=False)
        
        # Save in pickle format if requested (preserves object types like lists)
        if args.pickle_format:
            train_pickle_path = train_embed_path.replace('.csv', '.pkl')
            print(f"Saving training data with embeddings to {train_pickle_path}")
            original_train_df.to_pickle(train_pickle_path)
    
    # Only save test embeddings if explicitly requested
    # Test embeddings are often not needed for downstream tasks, so this is optional
    if args.save_test_embeddings:
        # Create the same text-to-embedding mapping for test data
        test_text_to_embedding = dict(zip(test_texts, test_embeddings_list))
        
        if args.text_column in original_test_df.columns:
            # Add embeddings to the test dataframe
            original_test_df['embedding'] = original_test_df[args.text_column].map(
                lambda x: test_text_to_embedding.get(x, None))
            
            # Save the updated original test dataframe with embeddings
            test_embed_path = os.path.join(run_dir, os.path.basename(args.test_path).replace('.csv', '_with_embeddings.csv'))
            print(f"Saving testing data with embeddings to {test_embed_path}")
            original_test_df.to_csv(test_embed_path, index=False)
            
            # Save in pickle format if requested
            if args.pickle_format:
                test_pickle_path = test_embed_path.replace('.csv', '.pkl')
                print(f"Saving testing data with embeddings to {test_pickle_path}")
                original_test_df.to_pickle(test_pickle_path)
    
    # Save raw embeddings as numpy arrays if requested
    # This can be useful for downstream tasks that need the embeddings directly
    if args.save_embeddings:
        # Always save train embeddings
        np.save(os.path.join(run_dir, 'train_embeddings.npy'), train_embeddings)
        
        # Only save test embeddings if explicitly requested
        if args.save_test_embeddings:
            np.save(os.path.join(run_dir, 'test_embeddings.npy'), test_embeddings)
            print(f"Train and test embeddings saved as numpy arrays in {run_dir}")
        else:
            print(f"Train embeddings saved as numpy arrays in {run_dir}")
        
        # Also save the class mapping for label interpretation
        with open(os.path.join(run_dir, 'label_encoder.json'), 'w') as f:
            json.dump({'classes': list(mlb.classes_)}, f, indent=4)
    
    # Save subset of features if dimensionality reduction was used
    if args.save_subset_features > 0:
        np.save(os.path.join(run_dir, 'train_embeddings_subset.npy'), train_embeddings[:, :args.save_subset_features])
        if args.save_test_embeddings:
            np.save(os.path.join(run_dir, 'test_embeddings_subset.npy'), test_embeddings[:, :args.save_subset_features])
            print(f"Subset of train and test embeddings saved as numpy arrays in {run_dir}")
        else:
            print(f"Subset of train embeddings saved as numpy arrays in {run_dir}")
    
    # Find similar requests for each test sample using cosine similarity
    print("\nFinding similar requests...")
    similar_indices, similarity_scores = find_similar_requests(
        test_embeddings, train_embeddings, train_labels, top_k=args.top_k
    )
    
    # Calculate label-based metrics to evaluate similarity quality
    print("\nCalculating label-based metrics...")
    k_values = [1, 3, 5, 10]  # Calculate precision@1, @3, @5, and @10
    label_metrics = calculate_label_based_metrics(test_labels, train_labels, similar_indices, k_values)
    
    # Print metrics for quick assessment
    print("\nLabel-based Metrics:")
    print(f"Match Rate (at least one match): {label_metrics['match_rate']:.4f}")
    for k in k_values:
        print(f"Average Precision@{k}: {label_metrics[f'avg_precision@{k}']:.4f}")
        print(f"Average Recall@{k}: {label_metrics[f'avg_recall@{k}']:.4f}")
        print(f"Average F1@{k}: {label_metrics[f'avg_f1@{k}']:.4f}")
    print(f"Average Label Overlap: {label_metrics['avg_label_overlap']:.4f}")
    
    # Visualize metrics at different k values
    plot_precision_at_k(label_metrics, k_values, run_dir)
    
    # Save results in JSON format for later analysis
    results = {
        'text_column': args.text_column,
        'similar_requests': {
            'indices': [indices.tolist() for indices in similar_indices],
            'scores': [scores.tolist() for scores in similarity_scores]
        },
        'label_metrics': label_metrics,
        'sbert_training': {
            'train_losses': train_losses if args.training_epochs > 0 else [],
            'test_losses': test_losses if args.training_epochs > 0 else [],
            'best_loss': float(best_loss) if args.training_epochs > 0 else None
        }
    }
    
    # Save detailed similarity results with label matching information
    # This provides a rich dataset for analysis of individual test cases
    similarity_results = []
    for i, (indices, scores) in enumerate(zip(similar_indices, similarity_scores)):
        # Create entry for each test sample
        test_sample = {
            'test_text': test_texts.iloc[i],
            'test_labels': test_labels[i],
            'similar_requests': []
        }
        
        # Convert test labels to set for faster intersection operations
        test_labels_set = set(test_labels[i])
        
        # Add information about each similar training sample
        for j, (idx, score) in enumerate(zip(indices, scores)):
            train_labels_set = set(train_labels[idx])
            matching_labels = list(test_labels_set & train_labels_set)  # Find common labels
            
            similar_request = {
                'rank': j + 1,
                'text': train_texts.iloc[idx],
                'labels': train_labels[idx],
                'similarity_score': float(score),
                'matching_labels': matching_labels,
                'has_matching_label': len(matching_labels) > 0
            }
            test_sample['similar_requests'].append(similar_request)
        
        similarity_results.append(test_sample)
    
    # Write the detailed similarity results to a JSON file
    with open(os.path.join(run_dir, 'similarity_results.json'), 'w') as f:
        json.dump(similarity_results, f, indent=4)
    
    # Write the summary results to a JSON file
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAnalysis completed! Results saved to {run_dir}")
    
    # Return a dictionary with important objects for further analysis if needed
    return {
        'model': model,
        'train_embeddings': train_embeddings,
        'test_embeddings': test_embeddings,
        'similar_indices': similar_indices,
        'similarity_scores': similarity_scores,
        'label_metrics': label_metrics,
        'results_dir': run_dir,
        'original_train_df': original_train_df,  # Return the original dataframe with embeddings
        'original_test_df': original_test_df if args.save_test_embeddings else None  # Return the test dataframe only if requested
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare change requests using SBERT embeddings and cosine similarity')
    
    parser.add_argument('--train_path', type=str, 
                        default="/kaggle/input/kubernetes-final-bug-data/train_data.csv",
                        help='Path to the training CSV data file')
    parser.add_argument('--test_path', type=str, 
                        default="/kaggle/input/kubernetes-final-bug-data/test_data.csv",
                        help='Path to the testing CSV data file')
    parser.add_argument('--text_column', type=str, default='all_text',
                        help='Column name with the text data to use for training and testing')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    
    parser.add_argument('--min_label_freq', type=int, default=1,
                        help='Minimum frequency for a label to be considered in training data')
    parser.add_argument('--max_label_len', type=int, default=100,
                        help='Maximum number of labels per sample')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generating embeddings')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading (increase for faster data loading)')
    parser.add_argument('--training_epochs', type=int, default=20,
                        help='Number of epochs for SBERT fine-tuning (0 to skip)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of similar requests to find for each test sample')
    
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Whether to save embeddings as separate files')
    parser.add_argument('--pickle_format', action='store_true',
                        help='Whether to also save dataframes with embeddings in pickle format (preserves the object types)')
    parser.add_argument('--save_subset_features', type=int, default=0,
                        help='Save only a subset of embedding features (e.g., 128). Default 0 means save all')
    parser.add_argument('--save_test_embeddings', action='store_true',
                        help='Whether to save test embeddings (disabled by default since usually not needed)')
    
    args, unknown = parser.parse_known_args()
    
    try:
        # Print diagnostic information
        print(f"Starting SBERT similarity analysis with parameters:")
        print(f"  Train data: {args.train_path}")
        print(f"  Test data: {args.test_path}")
        print(f"  Text column: {args.text_column}")
        print(f"  Min label frequency: {args.min_label_freq}")
        print(f"  Max label length: {args.max_label_len}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Training epochs: {args.training_epochs}")
        print(f"  Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        
        # Run the main function with error handling
        results = main(args)
        print("Analysis completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"\nERROR: An exception occurred during execution:")
        print(f"{type(e).__name__}: {str(e)}")
        if args.debug:
            print("\nDetailed traceback:")
            traceback.print_exc()
        print("\nTry adjusting parameters like min_label_freq or max_label_len to ensure sufficient data remains after filtering.")
        print("Use --debug flag for more detailed error information.")