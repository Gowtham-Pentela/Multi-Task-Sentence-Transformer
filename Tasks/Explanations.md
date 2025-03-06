### Task 1: Sentence Embedding Evaluation using DistilBERT

#### Step 1: Library Imports and Loading Pretrained Models
I began with importing necessary libraries like `torch`, `transformers`, and `datasets` to handle the DistilBERT model. I have included libraries to compute cosine similarity and PCA (Principal Component Analysis) to project embeddings to a lower dimension for visualization.

#### Step 2: Loading Data
I used the `bookcorpus` dataset from Hugging Face with a 1% sample for faster exploration. I loaded in the dataset to access text data, which I could then encode into embeddings using DistilBERT.

#### Step 3: Tokenization and Sentence Embeddings
One crucial part of this process was writing the function `encode_sentences`. This function tokenized text input using DistilBERT’s tokenizer and converted tokenized sentences to embeddings. I utilized **mean pooling** (i.e., average token embeddings from the last hidden state of the model) to generate fixed-size representations for a sentence. Token-level embeddings are not directly helpful for tasks like similarity or classification, so I was trying to obtain a compact sentence embedding.

I have tried out this function using sample sentences from the dataset and have printed out the resulting embeddings to confirm that the model was generating useful embeddings for text.

#### Step 4: Compute Cosine Similarity
To assess the quality of such embeddings, I quantified how similar certain pairs of sentences were to one another. For instance, I quantified similarity between similar sentiment-bearing sentences like "I love programming." and "I enjoy coding." using **cosine similarity**. I was expecting a high similarity score between them as they conveyed similar meanings.

Besides, I contrasted unrelated pairs such as "I love programming" and "The weather is nice today." This revealed that cosine similarity varied with semantic relationship between sentence pairs, which established that embeddings were effective for semantic similarity tasks.

#### Step 5: Query and Corpus Similarity
Carrying on from the previous task, I tested how well embeddings could match a query to a matching sentence in a corpus. Comparing query sentences with possible matching sentences in terms of their cosine similarities, I determined how well sentence embeddings captured relationships in varying contexts.

#### Step 6: STS-B (Semantic Textual Similarity Benchmark) Evaluation

I used **STS-B dataset** from GLUE benchmark to further assess the performance of the model. I encoded sentence pairs from the dataset, calculated their cosine similarities, and compared them with ground truth similarity scores.

To evaluate performance, I calculated **Mean Squared Error (MSE)** between actual and predicted similarity. Lower MSE meant that the predicted similarity was closer to human-assigned similarity and hence better model performance.

#### Step 7: Dimensionality Reduction for Visualization
I applied **PCA** (Principal Component Analysis) to reduce sentence embeddings to a lower dimension for visualization. This is a typical method for visualizing high-dimensional data in 2D. I visualized the embeddings in 2D and created an interactive scatter plot to observe how well embeddings clustered by semantic content.

#### Step 8: Model Evaluation: Sentence Length Encodings and Pooling Strategy

Several design decisions affected the quality of the embeddings:

a. **Pooling Strategy (Mean Pooling)**: I utilized mean pooling to pool token embeddings to create a sentence embedding with a specific size after transforming sentences using the transformer model.

b. **Dimensionality Reduction (PCA)**: I applied PCA to project sentence embeddings to 2D for visualization, making sentence relationships easy to explore.

c. **Maximum sentence length (max_length)**: I limited sentence length to 128 tokens to manage memory and computational complexity. This `max_length` value can be adjusted as per specific requirements.

---

### Task 2: Multi-Task Learning with IMDb Sentiment Classification

#### Step 1: Loading and Data Preprocessing
First, in Task 2, I downloaded IMDb dataset from Hugging Face and converted it to a pandas DataFrame for easy management. I have renamed columns for better readability and remapped sentiment labels (0 for negative and 1 for positive) to have a consistent labeling across the dataset.

#### Step 2: Multi-Task Learning Model Architecture
In multi-task learning, I have established a custom model class named `MultiTaskSentenceTransformer` that is a subclass of `torch.nn.Module`. This model is constructed with DistilBERT as a base encoder and has two modes: sentence embedding generation and sentiment classification. I have included a linear layer on top of DistilBERT for sentiment classification.

#### Step 3: DataLoader Creation
I have developed a custom class named `IMDBDataset` to handle tokenization and data preparation. This class tokenizes text using the DistilBERT tokenizer and prepares training inputs.

I then created **DataLoader** instances for training and test sets. DataLoaders enabled data shuffling and batching, which made training efficient.

### Step 3: DataLoader Creation
I have developed a custom IMDBDataset class to handle tokenization and data preparation. This class tokenizes text with the DistilBERT tokenizer and prepares training inputs.

I then defined DataLoader objects for both training and test sets. Both DataLoaders batched and shuffled data to enhance training efficiency.

### Step 4: Model Training
I started training with AdamW optimizer and CrossEntropyLoss for classification. For quicker training and reduced memory usage, I have turned on Automatic Mixed Precision (AMP) to use while training with GPUs and large datasets.

I tracked accuracy per epoch during training to see how well the model was performing in sentiment classification. I also used gradient accumulation to handle larger batches without crossing memory limits.

### Step 5: Model Evaluation
I then created an evaluate_model function to evaluate how well the model was doing on the test set. I computed significant measures like accuracy, precision, recall, and F1-score to gauge the performance of the model in sentiment classification.

### Step 6: Sentiment Analysis
Finally, I have defined a function predict_sentiment to predict a sentence's sentiment using a trained model. This function tokenizes a sentence and then outputs a sentiment prediction (either positive or negative).

### Task 3: Implications and Advantages of Freezing Layers in Multi-Task Learning

#### Key Decisions and Insights Overview:
In multi-task learning (MTL), whether to freeze or unfreeze is a decision that has a major bearing on how well a model learns to fit tasks. Below are the conclusions drawn from the three scenarios:

#### Freezing the Entire Network:
This is not usually advisable for tasks quite different from those for which the pre-trained model was initially developed. While it prevents overfitting and speeds training, model performance will be restricted to what is contained in pre-trained weights and will not generalize to new tasks. It is helpful when new tasks share a high level of similarity with tasks for which the model was pre-trained.

#### Freezing Only the Transformer Backbone:
Freezing only the backbone is a good balance. General knowledge from pre-training is retained in the backbone, and task-specific heads are trained to specialize in new tasks. This is highly efficient when overall language ability in a model is good but tasks have to be adapted at output level (e.g., classification, regression).

#### Freezing Only One Task-Specific Head:
This technique allows for a single task to be given priority and to reduce catastrophic forgetting for the frozen task. It is most helpful in cases where a single task is more important or has a larger amount of data than the other, so that the model can easily adapt to the more complex or important task without sacrificing on the performance of the other.

Practically, fine-tuning upper layers with frozen lower layers preserves the overall knowledge in the pre-trained model while still enabling upper-layer adaptations to specific tasks. This is a good balance between efficiency in computation and learning for specific tasks. The overall idea is to freeze lower layers (general features) and unfreeze upper layers and task heads to optimize for specific tasks.

---
### **Multi-Task Learning (MTL) Training Framework: Assumptions and Decisions**

Multi-Task Learning, or MTL, involves training a single model to handle several related tasks at the same time. The main advantage of this approach is that tasks can share information, which helps the model to perform better overall. Here, we'll explore key aspects of setting up MTL, focusing on **data management, how the model processes data, and how we evaluate success**.

---

## **1. Handling of Data**
### **Assumptions:**
- The model handles multiple tasks that can benefit from shared learning.
- The data includes **input features (`X`)** and different labels for each task (like `Y_task1`, `Y_task2`, etc.).
- Some tasks involve **classification** (e.g., sorting data into categories like determining sentiment), while others involve **regression** (e.g., predicting numerical values like sentiment strength).
- Tasks may have different amounts of data available, which can affect their weight in the learning process.

### **Decisions:**
- **Shared Feature Representation:** Instead of creating separate models for each task, one unified model part (like a common neural network) is used to learn features applicable to all tasks.
- **Task-Specific Heads:** Separate sections of the model are designed for each task to make accurate predictions.
- **Data Sampling Strategy:**
  - If tasks have different numbers of data samples, a strategy might be needed to ensure fair training.
  - This could involve **proportional sampling** (allocating batches based on the size of each task's data) or **round-robin sampling** (taking turns sampling from each task).

---

## **2. Forward Pass in MTL**
### **Assumptions:**
- The input `X` is processed through **shared layers** to extract useful features.
- These layers must be broad enough to cover all tasks but also allow for specific details needed by each task.
- Each task has a **task-specific head** that uses the shared features to make predictions.

### **Decisions:**
1. **Architecture Design:**
   - The model's shared part learns **features valuable** across all tasks.
   - After this shared section, the model splits into **task-specific heads**, each dedicated to a particular task.

2. **Task-Specific Specialization:**
   - Some tasks might require additional layers beyond the shared section.
   - The complexity of these additional parts can be adjusted according to each task's needs.

3. **Task Dependencies & Gradients:**
   - If one task is significantly harder, its learning signals might dominate others.
   - Techniques like **Gradient normalization** ensure balanced learning across tasks.

---

## **3. Metrics & Loss Computation**
### **Assumptions:**
- Each task has its own criteria for evaluating success:
  - **Classification tasks** use metrics like accuracy or F1-score.
  - **Regression tasks** use metrics like mean squared error (MSE).
- The overall success of the model depends on its performance across **all tasks**.

### **Decisions:**
1. **Loss Function Selection:**
   - For classification tasks, **Cross-Entropy Loss** is used to assess how well the model sorts data into categories.
   - For regression tasks, **Mean Squared Error (MSE)** is used to measure the difference between predicted and actual values.

2. **Loss Weighting Strategy:**
   - If one task's error consistently outweighs others, it might dominate the training process.
   - A **weighted loss function** is used to balance this, scaling each task's error appropriately.
   - Task weights can be:
     - **Fixed:** Set based on prior understanding of task importance.
     - **Dynamic:** Adjusted during training to reflect task difficulty.

3. **Multi-Task Performance Evaluation:**
   - Performance is measured using **individual task metrics** and overall performance.
   - A single model might not excel equally in all tasks, requiring trade-offs.
   - Significant drops in a task's performance might signal **negative transfer**, indicating a need for adjustments in the model or task balancing.