# GNNExplainer

**GNNExplainer** is a tool designed to explain the predictions made by Graph Neural Networks (GNNs). It helps interpret how GNNs make decisions, especially in graph-structured data, by identifying important subgraphs and node features. This is particularly useful for improving the transparency and trustworthiness of GNN models in fields like healthcare, finance, and social networks.

---

## Key Features:
- **Model-Agnostic:** Works with any GNN architecture without modifications.
- **Versatile:** Supports tasks like Node Classification, Link Prediction, and Graph Classification.
- **Single-Instance and Multi-Instance Explanations:** Explains individual predictions or provides global insights across multiple instances.
- **Optimization-Based:** Uses optimization techniques to isolate critical components of the graph and features that influence predictions.

---

## Why GNNExplainer?
Graph Neural Networks (GNNs) are powerful tools for analyzing graph-structured data, but their decision-making process can be difficult to interpret. Understanding how GNNs make predictions is crucial, especially in sensitive applications like social networks, healthcare, or finance.

### How GNNExplainer Works:
- **Subgraphs:** It zooms in on the most relevant subgraphs that influence a specific prediction.
- **Node Features:** It identifies the key node features that affect the model's decision.
- **Optimization Techniques:** It isolates the most important components while ignoring the less important ones, enhancing the transparency of GNN predictions.

---

## Flexible and Broadly Applicable
GNNExplainer is designed to be:
1. **Model-Agnostic:** Works with any Graph Neural Network (GNN) architecture.
2. **Versatile:** Supports tasks like Node Classification, Link Prediction, and Graph Classification.
3. **Single-Instance and Multi-Instance Explanations:**
   - **Single-Instance Explanations:** Focuses on explaining predictions for individual instances.
   - **Multi-Instance Explanations:** Provides insights across a group of instances, summarizing shared patterns.

---
## Use Case: Molecular Property Prediction

GNNExplainer is useful for applications like predicting molecular properties, where graph-structured data (such as atoms and bonds) is processed by a GNN model. For example, in molecular datasets like **QM9**, GNNExplainer can identify the critical subgraphs and features that contribute to the model's predictions.

### Ring Property Enhancement:
- **Purpose:** Adds ring classification (homocyclic or heterocyclic) to the molecular dataset to improve prediction models.
- **Processing:** Converts SMILES strings into RDKit molecules, classifies rings, and updates target property tensors.
- **Output:** Enhances the dataset with additional features for better model predictions.

---

## Dataset Splitting & DataLoader Setup
The dataset is split into training, validation, and test sets using **stratified sampling** to ensure class balance. The following code shows how the dataset is split:

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

## GNN Model Architecture (GCN & GAT)
The GNN models such as **GCN** (Graph Convolutional Network) and **GAT** (Graph Attention Network) can be trained to predict binary properties. The architectures include:

- **Multiple layers with dropout** for regularization.
- **Global Mean Pooling:** Aggregates node features for graph-level predictions.

---

## Model Training & Evaluation
The model is trained using **Binary Cross-Entropy Loss** and **Adam Optimizer**. The training process is evaluated using **validation accuracy**, and the best model is saved for further use. Here is an overview of the training and evaluation process:

### Training:
- Use the **Adam optimizer** with a learning rate of 0.001.
- The model is trained for a specified number of epochs, and **training loss** is tracked.

### Evaluation:
- After each epoch, the model is evaluated on the **validation set** to monitor performance.
- The **best model** is saved, and metrics like **accuracy** are plotted for visualization.

---

## Additional Information for Readers

### Understanding Graph Neural Networks (GNNs)
Graph Neural Networks (GNNs) are a class of neural networks designed to handle data represented as graphs. Graphs are structures that consist of nodes (or vertices) and edges (or links) that connect pairs of nodes. GNNs are particularly useful for tasks where the relationships between entities are as important as the entities themselves.

### Why Explanations Matter
In many real-world applications, especially those involving critical decisions (e.g., drug discovery, fraud detection), it's not enough for a model to be accurate—it must also be interpretable. GNNExplainer provides insights into why a GNN made a particular prediction, which can help users trust the model and understand its limitations.

### Applications of GNNExplainer
- **Healthcare:** Interpret predictions in drug discovery, protein interaction networks, and patient diagnosis.
- **Finance:** Understand fraud detection models and credit scoring systems.
- **Social Networks:** Analyze influence propagation, community detection, and recommendation systems.

---

## Getting Started with GNNExplainer
To get started with GNNExplainer, follow these steps:

1. **Install the necessary dependencies** as listed above.
2. **Prepare your dataset:** Ensure your data is in a graph format (nodes, edges, and features).
3. **Train your GNN model:** Use a GNN architecture like GCN or GAT.
4. **Apply GNNExplainer:** Use the tool to generate explanations for your model's predictions.

## Python Dependencies
To install the necessary libraries, use the following requirements file:

```bash
torch==2.5.1
torchvision
torchaudio
torch_geometric
rdkit==2024.03.6
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
pandas==2.2.3
pytorch-lightning
networkx==3.2.1

