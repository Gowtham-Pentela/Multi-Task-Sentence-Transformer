# Multi-Task Sentence Transformer for Sentiment and Semantic Similarity Analysis

This guide is about a Python program called a **Multi-Task Sentence Transformer**. The program uses a model named **DistilBERT** to help us understand sentences better. It can do tasks like finding out if a sentence is positive or negative, and checking how similar two sentences are in meaning. This project uses a special tool called the Hugging Face `transformers` library. The main tasks in this project include:

- Extracting information from sentences to check their similarity.
- Analyzing sentences from IMDb to determine if they are positive or negative.
- Performing multiple tasks such as understanding sentences and classifying them.

---

## Key Features

- **Sentiment Analysis**: The program can decide if a sentence expresses a positive or negative feeling using data from IMDb.
- **Semantic Similarity**: It calculates how similar two sentences are by using cosine similarity.
- **Multi-Task Learning**: The model can handle different tasks like creating sentence details and classifying them.
- **Dimensionality Reduction & Visualization**: It uses a method called PCA to make viewing sentence details easier.
- **Pre-trained DistilBERT**: The program uses a ready-to-use DistilBERT model for understanding sentences.

---

## System Requirements

- Python version 3.8 or newer
- PyTorch version 1.10 or newer
- Hugging Face `transformers` library
- `datasets` library
- Plotly for creating visual graphs
- scikit-learn for performance metrics

Install these requirements by running:

```bash
pip install -r requirements.txt
```

---

## Datasets

The project uses the following datasets:

- **BookCorpus**: Used for evaluating sentence details.
- **IMDb**: Used for identifying sentiment (positive or negative).
- **STS-B**: Used for evaluating semantic similarity between texts.

---

## Getting Started

### 1. Clone the project repository to your computer.

### 2. Install the required programs:

```bash
pip install -r requirements.txt
```

### 3. Execute the Task.ipynb file to start the tasks.

---

## Training Information

- **Model**: Pre-trained DistilBERT
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss for classification tasks

---

## Evaluation Metrics

- **Accuracy**: Percentage of correctly guessed outcomes.
- **Precision**: Correctness of positive predictions.
- **Recall**: Ability to find all positive instances.
- **F1-Score**: Balance of precision and recall for positive predictions.

---

## Visualizing Results

After training, you can visualize sentence details in 2D using **PCA**. This helps in understanding how well the sentence details are spread out and related to each other.

---

## How to Contribute

You can help improve this project by:

- Reporting any issues or bugs
- Suggesting new features or improvements
- Enhancing the code and submitting your changes

---

## License Information

This project is available under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

## Acknowledgements

Special thanks to:

- [Hugging Face](https://huggingface.co/) for the `transformers` library and pre-trained models.
- [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for providing sentiment data.
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for helping with dimensionality reduction and visualization.

---

## Contact Information

For questions or suggestions, please open an issue or reach out via email at gowthampentela@outlook.com