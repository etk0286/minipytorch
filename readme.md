# MiniPyTorch v2.0: DAG-Based Autograd and Neural Network Engine

## 📖 Project Overview
This project is an advanced implementation for a "Data Structures and Algorithms" course. Unlike traditional CRUD applications, this project dives straight into the core mechanics of modern AI frameworks (such as PyTorch and TensorFlow). 

Built entirely from scratch **without relying on any external deep learning libraries**, it is a lightweight tensor computation engine supporting **Automatic Differentiation (Autograd)**. 

In **v2.0**, the system evolves from a static forward-pass calculator into a complete AI engine capable of "learning." By implementing the calculus chain rule and Gradient Descent, we successfully simulate the full lifecycle of a neural network, allowing the model to automatically converge and discover hidden physical rules from raw data.

## ✨ v2.0 New Features & Highlights
1. **Backpropagation Engine**: Implemented a "time-reversal" traversal in the Directed Acyclic Graph (DAG) to automatically compute gradients for all nodes based on the forward execution sequence.
2. **Calculus Chain Rule Implementation**: Expanded the `Node` classes with dedicated partial derivative logic for `Add`, `Multiply`, `ReLU` activation, and `MSELoss` (Mean Squared Error).
3. **Gradient Descent & Training Loop**: Introduced a complete Epoch-based training mechanism where the model dynamically updates its weights ($w$) and biases ($b$) using computed gradients.
4. **Lossless Graph Reusability**: Solved Kahn's Algorithm's node-consumption bottleneck by introducing an "Indegree Shadow Copy" mechanism. The computational graph can now be executed infinitely without destroying its original topological structure.
5. **Training Process Visualization**: In addition to the topological DAG plots, v2.0 introduces Matplotlib-based Loss Convergence Curves to intuitively track the model's learning trajectory.

## 🎯 Core Data Structures & Algorithm Design
The essence of this project lies in transforming abstract mathematical processes into concrete data structure implementations:

| Data Structure / Algorithm | Application in Project | Engineering Bottleneck Solved |
| :--- | :--- | :--- |
| **Directed Acyclic Graph (DAG)** | Tracks dependencies between variables and computational nodes (Add, Multiply, ReLU, etc.). | Replaces rigid sequential code, transforming static math formulas into dynamically trackable, parallel-friendly topological structures. |
| **Queue (Deque)** | Implements Kahn's Algorithm for Topological Sorting (Forward Pass). | Reduces node scheduling time complexity to $\mathcal{O}(V+E)$, ensuring downstream nodes are triggered only when all prerequisites are met. |
| **Hash Table (Dictionary)** | Creates a "shadow copy" of node Indegrees. | Ensures that the original graph topology is not consumed or destroyed during the thousands of iterations in a machine learning training loop. |
| **Dynamic Array (List)** | Records the execution sequence and historical Loss metrics. | Perfectly fulfills the LIFO (Last-In-First-Out) traversal requirements essential for Backpropagation. |

## 📂 System Architecture & File Structure
The project adopts an Object-Oriented Programming (OOP) design, strictly adhering to the separation of concerns:

```text
minipytorch/
│
├── core/                  # Core Computation Engine
│   ├── node.py            # Defines the base Node class, math operations (Add, Mul), neural network layers (ReLU, MSELoss), and their derivative rules.
│   └── graph.py           # DAG Manager: handles edge connections, topological scheduling (execute), and backpropagation (backward).
│
├── utils.py               # Visualization tools utilizing NetworkX and Matplotlib to render DAG topologies.
├── main.py                # System entry point: defines model architecture, loads data, executes the training loop, and plots results.
├── environment.yml        # Conda environment configuration.
└── README.md              # Project documentation.
```
## 🚀 Quick Start

### 1. Environment Setup
This project uses Conda for environment management to ensure clean dependency tracking:

# Create a virtual environment
```bash
conda env create -f environment.yml
```

# Activate the environment
```bash
conda activate minipytorch
```

# Download the Dataset (Windows)
```bash
curl -L -o fashion-mnist_train.csv.zip [https://github.com/skilfoy/datasets/raw/main/fashion-mnist_train.csv.zip](https://github.com/skilfoy/datasets/raw/main/fashion-mnist_train.csv.zip)
tar -xf fashion-mnist_train.csv.zip
```
#
```

### 2. Run the Training Engine
Execute the main script in the project root directory to start the neural network training loop:
```bash
python main.py
```

## 📊 Visualizations & Execution Results
During execution, the program will sequentially display three levels of visualization, perfectly reproducing the learning journey of the AI model:

1. **Pre-Training Topology (Before Training):**
   An interactive chart pops up showing the initial DAG topology of $y_{pred} = wx + b$. Node values are uncalculated (`?`) and weights are initialized randomly.
2. **Post-Training Topology (After Training):**
   After the terminal swiftly completes 100 epochs of gradient descent, a second chart displays the optimized network. You will observe the weights converging perfectly to match the hidden dataset rules (e.g., $w \approx 3$, $b \approx 2$).
3. **Loss Convergence Curve:**
   A line chart plotting Mean Squared Error (MSE) against Training Epochs, proving the model's healthy and steady convergence to a loss of zero.

---

**Instructor Note:** *This project was developed to demonstrate a profound understanding of Graph Theory, Queues, and dynamic memory scheduling, applied directly to the foundational mathematics of Artificial Intelligence.*