# Face Recognition Benchmark

A comprehensive benchmarking system for comparing state-of-the-art face recognition models (ArcFace and QMagFace) on the LFW dataset.

## Project Structure

```
├── data/                   # All datasets
│   ├── lfw/                # Original LFW images (e.g., from Kaggle)
│   └── lfw_sklearn/        # LFW images downloaded via scikit-learn (auto-download)
│   └── aligned_lfw/        # Preprocessed, aligned images (112x112)
├── models/                 # Downloaded PyTorch model weights (not tracked in git)
├── results/                # Metrics, confusion matrices, misclassified samples (not tracked in git)
│   ├── metrics/            # Performance metrics and comparison plots
│   ├── confusion_matrices/ # Confusion matrices visualizations
│   └── misclassified/      # Misclassified samples for error analysis
├── src/                    # All source code files
│   ├── preprocess.py       # Face detection and alignment
│   ├── models.py           # Model loading and embedding extraction
│   ├── benchmark.py        # Benchmarking and evaluation
│   ├── embeddings.py       # Embedding extraction logic
│   ├── evaluate.py         # Evaluation metrics and utilities
│   ├── utils.py            # Utility functions (e.g., ensure_dir, cosine_similarity)
│   ├── visualize.py        # Visualization utilities
│   └── main.py             # Main pipeline script
├── environment.yml         # Conda environment specification
├── requirements.txt        # pip requirements
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## Prerequisites

1. Python 3.6+
2. CUDA-compatible GPU (recommended for faster processing)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aylinaydincs/face-recognition-benchmark.git
   cd face-recognition-benchmark
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate face_benchmark
   ```
   
   Alternatively, you can use pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the LFW dataset:
   ## Downloading the LFW Dataset from Kaggle

      You can download the LFW dataset from Kaggle, which is often faster and more reliable than the original site.
      Follow these steps to download the dataset:
      1. **Install Kaggle CLI**:  
         Make sure you have the Kaggle CLI installed. If not, install it using pip:

         ```bash
         pip install kaggle
         ```   
      2. **Get your Kaggle API key**:
         - Go to your Kaggle account settings: https://www.kaggle.com/account
         - Click "Create New API Token". This will download a file called `kaggle.json`.
         - Place `kaggle.json` in your home directory under `~/.kaggle/`.
      
      3. **Download the LFW dataset**:
         ```bash
         kaggle datasets download -d jessicali9530/lfw-dataset -p data/
         ```

      4. **Extract the dataset**:
         ```bash
         unzip data/lfw-dataset.zip -d data/
         ```
         The images will be available in `data/lfw/`.
      5. **(Optional) Update your configuration**
         Make sure your scripts or configuration point to the correct directory, e.g. `data/lfw/`.
      **Reference:** 
      [LFW Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)


## Usage

### Complete Pipeline

To run the complete pipeline (preprocessing and benchmarking):

```bash
python src/main.py
```

### Step-by-Step Execution

1. Face Detection and Alignment:
   ```bash
   python src/main.py --step preprocess --detector insightface
   ```

2. Model Benchmarking:
   ```bash
   python src/main.py --step benchmark --models arcface qmagface
   ```

### Configuration Options

* `--arcface_emb_dir`: ArcFace embeddings directory (default: `results/arcface_embeddings`)
* `--qmagface_emb_dir`: QMagFace embeddings directory (default: `results/qmagface_embeddings`)
* `--lfw_pairs_path`: LFW pairs CSV path (default: `data/pairs.csv`)
* `--results_dir`: Results directory (default: `results/benchmark_sklearn`)
* `--img_root`: Image root directory (default: `data/lfw_sklearn`)
* `--qmag_model`: QMagFace model path (default: `models/magface_model.pth`)
* `--n_pairs`: Number of pairs for sklearn pairs (default: `3000`)
* `--seed`: Random seed (default: `42`)

> **Note:**  
> If you do not manually download the LFW dataset, you can simply run the main pipeline:
> 
> ```bash
> python src/main.py
> ```
> 
> If the dataset is not found in your `data/lfw_sklearn` directory, the script will automatically download the LFW images using scikit-learn and prepare them for you. These images can be used directly in the pipeline.

## Results

After running the benchmark, results will be available in the `results` directory:

1. **Metrics**:  
   - **Accuracy**: Overall correct predictions rate  
   - **F1 Score**: Harmonic mean of precision and recall  
   - **Precision**: Correct positive predictions out of all positive predictions  
   - **Recall**: Correct positive predictions out of all actual positives  
   - **MCC**: Matthews correlation coefficient  
   - **ROC AUC**: Area under the ROC curve  
   - **EER**: Equal Error Rate and its threshold  
   - **TPR@FAR=1e-3 / 1e-4**: True Positive Rate at low False Acceptance Rates  
   - **Optimal Threshold**: Threshold maximizing Youden’s J statistic  
   - **Confusion Matrix**: Visualization of model predictions

2. **Confusion Matrices**: Visualization of model predictions

3. **Misclassified Samples**: Examples of incorrect predictions for error analysis

## Example Benchmark Results

Below are example results for ArcFace and QMagFace on both the original LFW dataset and the scikit-learn LFW dataset:

| Model    | Dataset   | Accuracy | F1 Score | Precision | Recall | MCC    | ROC AUC | EER    | TPR@FAR=1e-3 | TPR@FAR=1e-4 | Optimal Threshold | Confusion Matrix         |
|----------|-----------|----------|----------|-----------|--------|--------|---------|--------|--------------|--------------|-------------------|-------------------------|
| ArcFace  | Original  | 0.9418   | 0.9396   | 0.9763    | 0.9056 | 0.8860 | 0.9642  | 0.0777 | 0.7236       | 0.5152       | 0.2645 (0.1951)   | [[2934, 66],<br>[283, 2716]]  |
| QMagFace | Original  | 0.9752   | 0.9745   | 0.9993    | 0.9510 | 0.9514 | 0.9772  | 0.0463 | 0.9510       | 0.9503       | 0.2858 (0.1439)   | [[2998, 2],<br>[133, 2866]]   |
| ArcFace  | sklearn   | 0.9520   | 0.9505   | 0.9811    | 0.9218 | 0.9057 | 0.9703  | 0.0613 | 0.7448       | 0.0004       | 0.2687 (0.2039)   | [[64726, 1173],<br>[5153, 60746]] |
| QMagFace | sklearn   | 0.9814   | 0.9810   | 0.9994    | 0.9633 | 0.9634 | 0.9851  | 0.0329 | 0.9634       | 0.9622       | 0.2788 (0.1556)   | [[65864, 35],<br>[2420, 63479]]  |

- **Original**: Results on the original LFW dataset (e.g., from Kaggle).
- **sklearn**: Results on the LFW dataset as downloaded and formatted by scikit-learn.

> **Note:**  
> Some LFW identities contain miscropped images (different people under the same name), which can affect results. Manual cleaning is recommended for best accuracy.  
> Both ArcFace and QMagFace benchmarks use the **ResNet100** architecture for fair comparison.

## Customization

* To add more face recognition models, extend the `FaceRecognitionModel` class in `src/models.py`
* For different evaluation metrics, modify the `evaluate_model` method in `src/benchmark.py`

## References

* ArcFace: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)  
  [ArcFace GitHub Repository](https://github.com/deepinsight/insightface)
* QMagFace: [QMagFace: Simple and Accurate Quality-Aware Face Recognition](https://openaccess.thecvf.com/content/WACV2023/papers/Terhorst_QMagFace_Simple_and_Accurate_Quality-Aware_Face_Recognition_WACV_2023_paper.pdf)  
  [QMagFace GitHub Repository](https://github.com/IrvingMeng/QMagFace)
* LFW Dataset: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
* InsightFace: [InsightFace: 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface)

> **Note:**  
> During benchmarking, it was observed that the LFW dataset contains some miscropped images—different people grouped under the same identity. This can negatively affect the results for both ArcFace and QMagFace. For best results, consider manually reviewing and eliminating such miscropped or mislabeled samples from the dataset.
>
> Both ArcFace and QMagFace benchmarks in this project use the **ResNet100** architecture for fair comparison.