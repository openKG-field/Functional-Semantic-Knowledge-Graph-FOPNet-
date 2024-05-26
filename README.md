# Functional-Semantic-Knowledge-Graph-FOPNet-
# Paper link
- https://www.sciencedirect.com/science/article/pii/S1751157723000925
# Project Overview

This project is focused on patent-oriented semantic matching. Research and experiments were conducted on:
- Direct similarity matching methods
- Deep matching models
- Semantic matching weight methods
- Similarity threshold settings
- Text weight allocation during the matching process

## Project Operation Instructions

Run `FOPproject/FOP/TextMatching_eng.py` to calculate patent pair similarity. Optional parameters include feature weights, similarity thresholds, and text allocation.

## Project Directory Tree

    FOP
    - `TextMatching_eng.py`
    - `FOP_eng.py`
    - `Statistics_eng.py`
    - `FileProcess_eng.py`
    - `Weight_eng.py`
    - `Formula2_eng.py`
    - `Make_input_data.py`
    - `BM25_eng.py`
    - `Extract_fop_withClient.py`
    - `MAP_MRR_NDCG_eng.py`
    - `Recall_at_x_eng.py` 
    FOPdata/data_FOP_eng
    - `eng_ipc_fop.txt`
    - `extraction_method_1_dict_eng.txt`
    - `extraction_method_1_eng.txt`
    - `vector.txt`
    Todolist
    - `deep_matching`
    - `args.py`
    - `deep_train.py`
    - `dssm_train.py`
    - `evaluate.py`
    - `graph.py`

## File Descriptions

### FOP
- `TextMatching_eng.py`
Function: The main file for similarity calculation. It transfers files and variable values to different calculation files and calls various calculation methods.

    - `Init()`: Initialize
    - `common_main()`: Run four vector-based matching methods
    - `main()`: Run all matching methods
    - `MAP_MRR_NDCG_main()`: Run values for MAP, MRR, and NDCG
    - `FOPExtracted`: Extracted FOP triplet file
    - `Wordvector`: Vector file

- `FOP_eng.py`
Main function: Calculate the FOP similarity of each pair of samples and store it in an Excel document.

    - `FOPLabel`: Pass in a statement and determine whether it is the first half of the claim
    - `splitFOP`: Pass in string form FOP and return array form FOP
    - `splitKnowledge`: Pass in knowledge pairs in string form and return knowledge pairs in array form
    - `init`: Initialize variables according to calculation requirements
    - `format_2`: Processing samples, storing FOP into three separate arrays, calculating the vector of FOP, and calculating the variance in each document
    - `new_all`: Main function to calculate the FOP similarity between a pair of samples
    - `similarity_FOP`: Calculate the similarity between a pair of phrases
    - `similarity_2words`: Calculate the similarity between a pair of phrases using different methods
    - `get_FOP_sim`: Calculate the similarity between a pair of FOPs using formula 1
    - `get_patent_sim`: Calculate the similarity of a pair of samples using the sum of FOP similarities/FOP quantity
    - `get_sim_w_weight`: Calculate the similarity of a pair of samples using different weight methods
    - `get_sim_w_thre`: Calculate the similarity of a pair of samples using different thresholds
    - `get_sim_w_textWeight`: Calculate the similarity of a pair of samples using different weights for front and back parts
    - `Common_main`: General equation for calculating sample similarity using only vector methods
    - `main`: Total equation for calculating sample similarity

- `Statistics_eng.py`
Function: Statistically calculate the mean, median, mode, and standard deviation of results.

- `FileProcess_eng.py`
Function: Initialize and convert FOP triplet files and vector files into dictionaries/arrays.

    - `to_dict`: Process sample pairs output as array type
    - `to_vec`: Process vectors, convert string type vectors to array types
    - `to_ipcid`: Process patent IPC and transform it into vector form
    - `get_grade`: Process the type rating between each pair of samples
    - `get_data`: Process sample pairs, integrating all the corresponding second samples of the first sample of each sample pair
    - `to_dict_w_pair`: Process sample pairs, output sample pair FOP, sample pair ID, and data for all samples

- `Weight_eng.py`
Main function: Provides calculation of different weight methods including BM25, tfidf, graph nodes, k-means aggregation, and spectral aggregation.

    - `__init__`: Initialization function
    - `set_up`: Prepare variables for weight calculation
    - `__bm25_set_up`: Set bm25 and return the bm25 value matrix
    - `__to_FOP`: Split all FOP strings into FOP arrays for BM25 calculation
    - `__to_word`: Break FOP into individual word combinations for BM25 calculation
    - `bm25`: Calculate BM25 weight
    - `KMeans`: Calculate KMeans weights
    - `SpectralClustering`: Calculate Spectral Clustering weights
    - `Graph`: Calculate Graph Weights
    - `tfidf`: Calculate tfidf weights

- `Formula2_eng.py`
Main functions: Similarity calculation formulas for different methods including:
    - Character matching: Dice coefficient, inclusion index, Jaccard coefficient
    - Vector matching: Euclidean distance, Pearson coefficient, Spearman coefficient, Cosine distance
    - Concept matching: Lin method, Resnik method, Jiang method

- `Make_input_data.py`
Function: Create input files.

- `BM25_eng.py`
Function: BM25 weight method calculation. Called by the `Weight_eng.py` file.

- `Extract_fop_withClient.py`
Function: Used for extracting FOP.

- `MAP_MRR_NDCG_eng.py`
Main function: Implement statistical results of MAP, MRR, and NDCG indicators.

    - `MAP_main`: Calculate the similarity for each pair of samples and then calculate metric results
    - `calculate`: Calculate sample similarity
    - `get_mean_score`: Calculate the average value of indicator results
    - `get_MRR_score`: Calculate the MRR results
    - `MAP_MRR_NDCG_main`: Main function

- `Recall_at_x_eng.py`
Main function: Implement recall statistics results.

    - `generate_random_100`: Randomly generate a main sample and its corresponding comparison sample
    - `write_dataset`: Generate result files
    - `formad_field_tech`: Generate field_tech data for sample pairs and calculate the number of occurrences of different field_tech combinations
    - `All_main`: Calculate the similarity of each pair of samples

### FOPdata/data_FOP_eng
Function: Experimental related data.

Files prefixed with `extract` contain extracted FOP triples but are stored in various formats.

- `eng_ipc_fop.txt`
Patent number and corresponding IPC classification.

- `extraction_method_1_dict_eng.txt`
Main function: Calculate recall using data format: Patent number: All triplets of the patent.

- `extraction_method_1_eng.txt`
Main functions: Matching, calculating similarity. Data format: Patent number: FOP triplet A.

- `vector.txt`
Main function: English vector dictionary.

### Todolist

   - `deep_matching`
   - `args.py`: Parameters used for DSSM training
   - `deep_train.py`: Main program for deep model training except for DSSM
   - `dssm_train.py`: DSSM training main program
   - `evaluate.py`: Calculate recall and metric results for deep learning
   - `graph.py`: DSSM network building diagram

## Data Description

Note: The data section only provides one thousand pairs of data for reference. Please contact the author for the complete version of the data. 
- email:wzy03051241@163.com
