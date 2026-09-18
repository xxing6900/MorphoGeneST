# MorphoGeneST
<img width="1826" height="1261" alt="image" src="https://github.com/user-attachments/assets/7264b565-4e6d-44ff-bd34-88b6a1cadf1a" />
Figure 1. Overall workflow of MorphoGeneST for morphology-guided spatial transcriptomics prediction. (1) Spot-aligned data construction by extracting H&E patches and preparing dual training targets for regression and ZINB distribution; (2) Network inference integrating pathology, gene, and spatial streams through large-kernel encoding, SAGT aggregation, and parallel MoE decoding; (3) Training optimization via a multi-task joint loss function encompassing MSE, ZINB, and consistency-based self-distillation loss ℒKD; and (4) Generation of outputs for diverse downstream analyses including spatial domain identification, differential marker analysis, TME scoring, and spatial pseudotime trajectory inference.
## 📂 Repository Structure

The codebase is organized as follows:

```text
MorphoGeneST/
├── data/                       # Directory for datasets (needs to be downloaded manually)
│   ├── her2st/                 # HER2+ Breast Cancer ST dataset
│   └── GSE144240_RAW/          # cSCC dataset
├── code/                       # Main source code directory
│   ├── main.py / main2.py      # Training scripts for HER2+ and cSCC datasets
│   ├── dataset.py              # PyTorch Dataloaders for processing H&E images & ST counts
│   ├── modules.py              # Core architectures (ConvNeXt, SAGT, MoE Decoder)
│   ├── zinb.py                 # Zero-Inflated Negative Binomial (ZINB) loss and activations
│   ├── eval.py / e.py          # Global evaluation scripts (PCC, RMSE, SSIM, ARI calculation)
│   ├── eval2.py / e2.py        # Evaluation scripts specific for the cSCC dataset
│   ├── analysis.py / m33.py    # Comprehensive downstream biological analysis pipelines
│   ├── test.py / top4gene.py   # Single-fold testing and top marker gene spatial visualization
│   └── utils.py / predict.py   # Helper functions, metrics, and spatial clustering tools
└── requirements.txt            # Python dependencies
        
    
MorphoGeneST 🔬
Morphology-Guided Spatial Transcriptomics Inference via Mixture-of-Experts and Distance-Aware Graph Transformers
MorphoGeneST is a morphology-guided deep learning framework designed to infer spatial transcriptomics (ST) directly from standard Hematoxylin and Eosin (H&E) stained histopathological images. By seamlessly integrating a ConvNeXt-based morphological encoder, a Distance-Aware Spatial Graph Transformer (SAGT), and a probabilistic Mixture-of-Experts (MoE) decoder, MorphoGeneST robustly preserves anatomical boundaries and accurately denoises technical sequencing dropouts (via ZINB modeling).

📂 Repository Structure
The codebase is organized as follows:

        
text

        

            

        

        
            
MorphoGeneST/
├── data/                       # Directory for datasets (needs to be downloaded manually)
│   ├── her2st/                 # HER2+ Breast Cancer ST dataset
│   └── GSE144240_RAW/          # cSCC dataset
├── code/                       # Main source code directory
│   ├── main.py / main2.py      # Training scripts for HER2+ and cSCC datasets
│   ├── dataset.py              # PyTorch Dataloaders for processing H&E images & ST counts
│   ├── modules.py              # Core architectures (ConvNeXt, SAGT, MoE Decoder)
│   ├── zinb.py                 # Zero-Inflated Negative Binomial (ZINB) loss and activations
│   ├── eval.py / e.py          # Global evaluation scripts (PCC, RMSE, SSIM, ARI calculation)
│   ├── eval2.py / e2.py        # Evaluation scripts specific for the cSCC dataset
│   ├── analysis.py / m33.py    # Comprehensive downstream biological analysis pipelines
│   ├── test.py / top4gene.py   # Single-fold testing and top marker gene spatial visualization
│   └── utils.py / predict.py   # Helper functions, metrics, and spatial clustering tools
└── requirements.txt            # Python dependencies
        
    
🛠️ Installation & Requirements
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (e.g., Conda). Install the required packages via:

        
bash

        

            

        

        
            
pip install -r requirements.txt
        
    
Key Dependencies:
torch >= 1.12.0
pytorch-lightning
scanpy & squidpy
anndata
gseapy
scikit-image, scikit-learn, scipy
pandas, numpy, matplotlib, seaborn
💾 Data Preparation
MorphoGeneST is validated on two comprehensive spatial transcriptomics cohorts. Please download them and place them in the ./data directory according to the dataloader paths in dataset.py:

HER2+ Breast Cancer (HER2ST):
Place under ./data/her2st/data/. Required subdirectories: ST-cnts, ST-imgs, ST-spotfiles, ST-pat/lbl.
Cutaneous Squamous Cell Carcinoma (cSCC):
Place under ./data/GSE144240_RAW/.
Ensure the highly variable gene (HVG) lists (her_hvg_cut_1000.npy and skin_hvg_cut_1000.npy) are placed directly in the ./data/ folder.

🚀 Usage
1. Training the Model
To train the MorphoGeneST model from scratch, execute the main training scripts. The model uses PyTorch Lightning with Automatic Mixed Precision (AMP) enabled.

For HER2+ Breast Cancer Dataset:
        
bash

        

            

        

        
            
python main.py
        
    
(You can modify the fold variable in main.py’s __main__ block to train specific Leave-One-Section-Out cross-validation folds).

For cSCC Dataset:
        
bash

        

            

        

        
            
python main2.py
        
    
Model checkpoints will be automatically saved in the specified checkpoint directories (e.g., ./model_checkpointsljx30).

2. Global Evaluation
To comprehensively evaluate the model across all cross-validation folds and calculate macro-metrics (Gene-wise PCC, Overall PCC, RMSE, SSIM, and Leiden ARI):

Evaluate HER2+ Cohort:
        
bash

        

            

        

        
            
python e.py
# Or use python eval.py for detailed reporting
        
    
Evaluate cSCC Cohort:
        
bash

        

            

        

        
            
python e2.py
# Or use python eval2.py
        
    
3. Comprehensive Downstream Analysis
MorphoGeneST comes with a powerful automated biological analysis pipeline (m33.py or analysis.py). This pipeline generates publication-ready figures characterizing the Tumor Microenvironment (TME).

Configure the TARGET_FOLD and CHECKPOINT_PATH at the bottom of the script, then run:

        
bash

        

            

        

        
            
python m33.py
        
    
The pipeline automatically executes:
Marker gene spatial quantification & visualization.
Dropout Recovery Plausibility: Validates imputed spatial zeros against physical neighborhoods.
Unsupervised spatial domain identification (Leiden clustering) & ARI metrics.
TME functional scoring (B-cells, CD8 T-cells, Tumor clones) & statistical significance testing (Kruskal-Wallis).
Differential Expressed Genes (DEGs) dotplots/heatmaps & GO/KEGG pathway enrichment.
Exploratory Spatial-State Ordering (Pseudotime trajectory inference).
MoE Expert routing interpretability visualization.
Results will be exported to localized output folders (e.g., ./analysis_results22/Fold_X), including .h5ad AnnData objects, quantitative .csv tables, and high-resolution .png/.pdf figures.

🧠 Architectural Highlights
LayerNormChannels & ConvNeXt: Achieves batch-contamination-free morphological extraction mapping diverse local tissues robustly.
Distance-Aware Spatial Graph Transformer (SAGT): Projects learned physical distance penalties into self-attention matrices to eradicate topological over-smoothing and strictly preserve anatomical boundaries.
Probabilistic MoE & ZINB Decoder: Discriminates technical sequencing dropouts from biological silence, dynamically recovering localized expression omissions natively.
📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

For any questions or issues regarding the code or datasets, please open an issue in this repository.
