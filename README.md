# GitHub Issue Classification using Machine Learning

This repository contains the code and experiments for classifying GitHub issues into bug reports and feature requests using various machine learning approaches.

## Project Overview

This project develops automated classification systems to distinguish between bug reports and feature requests in GitHub issues. The research explores multiple approaches including traditional NLP models and modern transformer-based architectures.

## Repository Structure

```
├── README.md
├── requirements.txt           # Python dependencies
├── .gitignore
├── Collect Data/             # Data collection notebooks and scripts
│   ├── get_issues.ipynb      # GitHub API issue collection
│   ├── get_pr.ipynb          # Pull request data collection
│   ├── check_pr.ipynb        # PR validation and analysis
│   └── data_exploration.ipynb
├── Data/                     # Raw and processed datasets
│   ├── Data From API/        # Raw data from GitHub API
│   │   ├── Kind:Bug/         # Bug-labeled issues
│   │   └── Kind:Feature/     # Feature request issues
│   └── Preprocessed Data/    # Cleaned and processed datasets
│       ├── kind:bug/
│       └── kind:feature/
├── kind:bug/                 # Bug classification pipeline
│   ├── 1_Data_Collection/    # Link PRs with issues, extract file changes
│   ├── 2_Data_Preparation/   # Text preprocessing and augmentation
│   ├── 3_explore_data/       # Data exploration and analysis
│   ├── 4_modelling/          # Model training and evaluation
│   │   ├── Model Comparison/ # Comparative analysis
│   │   └── other model/      # Alternative approaches
│   └── 5_similiarity_check/  # Similarity analysis experiments
│       ├── 1_using_sbert/    # SBERT-based similarity
│       ├── 2_using_fasttext/ # FastText embeddings
│       └── 3_DeBERTa -> SBERT/ # Hybrid approaches
└── kind:feature/            # Feature request classification pipeline
    ├── 1_Data_Collection/    # Feature-specific data collection
    ├── 2_Data_Preparation/   # Preprocessing with augmentation
    │   ├── Text Augmentation/
    │   └── text augmented without filename/
    └── 3_Data_Modelling/     # Feature classification models
```

## Key Features

- **Multi-approach Classification**: Implements various models including CodeBERT, DeBERTa, and FastText
- **Data Augmentation**: Text augmentation techniques to improve model performance
- **Similarity Analysis**: Advanced similarity checking using SBERT and FastText embeddings
- **Comprehensive Evaluation**: Multiple evaluation metrics and comparison studies

## Models Implemented

### Bug Classification

- CodeBERT with multi-label classification
- DeBERTa with CNN layers
- FastText embeddings for similarity matching
- SBERT for semantic similarity

### Feature Request Classification

- CodeBERT variations with different input features
- DeBERTa with data augmentation
- Multi-label classification approaches

## Getting Started

### Prerequisites

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
# Core libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# Deep learning and NLP
pip install torch transformers sentence-transformers fasttext

# Jupyter environment
pip install jupyter notebook ipywidgets

# Additional utilities
pip install tqdm nltk spacy plotly
```

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (optional, will auto-download when needed)
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   python -m spacy download en_core_web_sm
   ```

### Data Collection

1. Navigate to `Collect Data/` folder
2. Run `get_issues.ipynb` to collect GitHub issues via API
3. Use `get_pr.ipynb` to gather pull request data
4. Validate data with `check_pr.ipynb`
5. Raw data will be saved in `Data/Data From API/`

### Data Preprocessing

#### For Bug Classification:

1. Navigate to `kind:bug/2_Data_Preparation/`
2. Run notebooks in sequence:
   - `1_merge_data.ipynb` - Combine datasets
   - `2_get_comments.ipynb` - Extract issue comments
   - `3_text_preprocessing.ipynb` - Clean and preprocess text
3. Processed data saved in `Data/Preprocessed Data/kind:bug/`

#### For Feature Classification:

1. Navigate to `kind:feature/2_Data_Preparation/`
2. Run preprocessing notebooks
3. Use `feature_filter.ipynb` for data filtering
4. Apply text augmentation techniques in `Text Augmentation/`

### Model Training

#### Bug Classification Models:

1. Navigate to `kind:bug/4_modelling/`
2. Choose from available models:
   - `1_codebert-multi-label-bug.ipynb` - CodeBERT baseline
   - `5_deberta-cnn.ipynb` - DeBERTa with CNN layers
   - `3_deberta-with-removed-token-length-outlier.ipynb` - Optimized DeBERTa
3. Results saved automatically with timestamps

#### Feature Classification Models:

1. Navigate to `kind:feature/3_Data_Modelling/`
2. Run desired models:
   - `4_codebert-multi-label.ipynb` - Multi-label CodeBERT
   - `5_deberta-augment-feature-request.ipynb` - Augmented DeBERTa

#### Similarity Analysis:

1. Navigate to `kind:bug/5_similiarity_check/`
2. Choose similarity approach:
   - `1_using_sbert/` - Sentence-BERT embeddings
   - `2_using_fasttext/` - FastText similarity matching

## Key Notebooks

### Data Collection

- `Collect Data/get_issues.ipynb` - GitHub API issue collection
- `Collect Data/get_pr.ipynb` - Pull request data gathering
- `Collect Data/check_pr.ipynb` - Data validation and quality checks

### Bug Classification

- `kind:bug/4_modelling/1_codebert-multi-label-bug.ipynb` - CodeBERT implementation
- `kind:bug/4_modelling/5_deberta-cnn.ipynb` - DeBERTa with CNN architecture
- `kind:bug/5_similiarity_check/2_using_fasttext/2_a_fasttext_similiarity_check.py` - FastText similarity
- `kind:bug/3_explore_data/1_final_data_exploration.ipynb` - Data analysis

### Feature Classification

- `kind:feature/3_Data_Modelling/4_codebert-multi-label.ipynb` - Multi-label CodeBERT
- `kind:feature/3_Data_Modelling/5_deberta-augment-feature-request.ipynb` - Augmented DeBERTa
- `kind:feature/2_Data_Preparation/feature_filter.ipynb` - Feature-specific filtering

### Text Preprocessing

- `kind:bug/2_Data_Preparation/3_text_preprocessing.ipynb` - Bug text preprocessing
- `kind:feature/2_Data_Preparation/Text Augmentation/` - Data augmentation techniques

## Results

The project includes comprehensive evaluation approaches:

### Similarity Analysis

- **Cosine similarity** - Semantic text similarity measurements
- **Euclidean distance** - Vector space distance calculations
- **Jaccard similarity** - Binary feature overlap analysis
- **Precision@K, Recall@K, F1@K** - Ranking-based evaluations

### Model Performance

- **Multi-label classification** metrics for complex issue labeling
- **Cross-validation** results across different data splits
- **Comparative analysis** between transformer models (CodeBERT, DeBERTa)
- **Augmentation impact** studies on model performance

### Key Findings

- Transformer models significantly outperform traditional approaches
- Text augmentation improves performance on limited datasets
- Similarity-based approaches effective for related issue detection
- Combined title+body+comments provide best classification accuracy

## Data

The datasets include:

- GitHub issues from various repositories
- Pull request information with linked issues
- Preprocessed text with multiple cleaning approaches
- Augmented datasets for improved training

## Contributing

This is research code for academic purposes. The experiments demonstrate various approaches to automated issue classification in software development contexts.

## License

Academic research project - please cite if used in academic work.

## Contact

For questions about this research, please refer to the original thesis documentation.
