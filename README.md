# Change Request Analysis and Classification System

This repository contains the implementation and experiments for my undergraduate thesis research on automated change request analysis using machine learning and natural language processing techniques.

## Project Overview

This research project focuses on two main objectives:

1. **Area Module Classification for Change Requests** - Automatically classifying which area modules are affected by incoming change requests
2. **Semantic Similarity-based Change Request Recommendation System** - Developing a recommendation system that suggests similar change requests based on semantic content analysis

The project explores various machine learning approaches including traditional NLP models and modern transformer-based architectures (Bert-base, RoBERTa, CodeBERT, DeBERTa) with comprehensive experiments conducted both with and without filename features.

## Repository Structure

```
├── README.md
├── requirements.txt                   # Python dependencies
├── .gitignore
├── code buku skripsi/                 # Main thesis experiments and results
│   ├── 1_Klasifikasi Area Modul yang Terdampak pada Change Request/
│   │   ├── 1_Eksperimen menggunakan nama file/       # Experiments with filename features
│   │   └── 2_Eksperimen tanpa menggunakan nama file/ # Experiments without filename features
│   └── 2_Sistem Rekomendasi Change Request Berbasis Kemiripan Semantik/
│       ├── 1_Eksperimen menggunakan nama file/       # Similarity experiments with filenames
│       └── 2_Eksperimen tanpa menggunakan nama file/ # Similarity experiments without filenames
├── Collect Data/                      # Data collection notebooks and scripts
│   ├── get_issues.ipynb              # GitHub API issue collection
│   ├── get_pr.ipynb                  # Pull request data collection
│   ├── check_pr.ipynb                # PR validation and analysis
│   └── data_exploration.ipynb
├── Data/                             # Raw and processed datasets
│   ├── Data From API/                # Raw data from GitHub API
│   │   ├── Kind:Bug/                 # Bug-labeled issues
│   │   └── Kind:Feature/             # Feature request issues
│   └── Preprocessed Data/            # Cleaned and processed datasets
│       ├── kind:bug/
│       └── kind:feature/
├── kind:bug/                         # Bug classification pipeline
│   ├── 1_Data_Collection/            # Link PRs with issues, extract file changes
│   ├── 2_Data_Preparation/           # Text preprocessing and augmentation
│   ├── 3_explore_data/               # Data exploration and analysis
│   ├── 4_modelling/                  # Model training and evaluation
│   │   ├── etc/                      # Additional model experiments
│   │   ├── with filename/            # Models trained with filename features
│   │   ├── without filename/         # Models trained without filename features
│   └── 5_similiarity_check/          # Similarity analysis experiments
│   │   ├── etc/                      # Additional model experiments
│       ├── with filename/            # Similarity with filename features
│       └── without filename/         # Similarity without filename features
└── kind:feature/                     # Feature request classification pipeline
    ├── 1_Data_Collection/            # Feature-specific data collection
    ├── 2_Data_Preparation/           # Preprocessing with augmentation
    │   ├── Text Augmentation/
    │   └── text augmented without filename/
    ├── 3_Data_Exploration/           # Feature data analysis
    ├── 4_Data_Modelling/             # Feature classification models
    │   ├── etc/
    │   ├── with filename/
    │   └── without filename/
    └── 5_similiarity_check/          # Feature similarity analysis
        ├── with filename/
        └── without filename/
```

## Contact and Support

This research was conducted as part of an undergraduate thesis project. For questions regarding:

- **Technical Implementation**: Refer to inline code documentation and notebook explanations
- **Research Methodology**: Consult the thesis document and experiment notebooks
- **Data Access**: Follow the data collection procedures outlined in the notebooks
- **Model Reproduction**: Use the provided configuration files and training scripts
