# MiniPyTorch: DAG-Based Autograd and Neural Network Engine

## 📖 Project Overview
This project is the final deliverable for the "Data Structures" course. Unlike traditional foundational CRUD applications (e.g., management systems), this project dives straight into the core of modern AI frameworks (such as PyTorch and TensorFlow). 

Built entirely from scratch without relying on any external deep learning libraries, it is a lightweight tensor computation engine supporting **Automatic Differentiation (Autograd)**. The system simulates the complete lifecycle of a neural network, including graph-based dynamic scheduling, forward pass evaluation, and backward pass computation based on the chain rule of calculus. Using a single-layer perceptron as an example, the project successfully demonstrates how a model automatically converges and learns hidden physical patterns through Gradient Descent.

## 🎯 Core Data Structures & Algorithm Design
The essence of this project lies in transforming abstract mathematical and computational processes into concrete data structure implementations:

| Data Structure / Algorithm | Application in Project | Engineering Bottleneck Solved |
| :--- | :--- | :--- |
| **Directed Acyclic Graph (DAG)** | Tracks dependencies between variables and computational nodes (Add, Multiply, ReLU, etc.). | Replaces rigid sequential code, transforming static math formulas into a dynamically trackable and parallel-friendly topological structure. |
| **Queue (Deque)** | Implements Kahn's Algorithm for Topological Sorting. | Reduces the time complexity of node scheduling to $\mathcal{O}(V+E)$, ensuring downstream nodes are only triggered when all prerequisites are met. |
| **Hash Table (Dictionary)** | Creates a "shadow copy" of node Indegrees. | Ensures that the original topological structure is not destroyed or consumed during the thousands of iterations required in a machine learning training loop. |
| **Dynamic Array (List)** | Stores the execution sequence for "time-reversal" (Backward Pass) and historical Loss metrics. | Perfectly fulfills the LIFO (Last-In-First-Out) traversal requirements essential for Backpropagation. |

## 📂 System Architecture & File Structure
The project adopts an Object-Oriented Programming (OOP) design, strictly adhering to the separation of concerns:

```text
dag_project/
│
├── core/                  # Core Computation Engine
│   ├── node.py            # Defines the base Node class, specific math operations (Add, Multiply, ReLU, MSELoss), and partial derivative rules.
│   └── graph.py           # DAG Manager responsible for edge connections, topological scheduling (execute), and backpropagation (backward).
│
├── utils.py               # Visualization tools utilizing NetworkX and Matplotlib to render DAG topologies.
├── main.py                # System entry point: defines model architecture, loads training data, and executes the gradient descent loop.
└── README.md              # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup
This project uses Conda for environment management to ensure clean dependency tracking:
```bash
# Create a virtual environment
conda create -n dag-engine python=3.10 -y

# Activate the environment
conda activate dag-engine

# Install visualization and scientific computing dependencies
conda install numpy networkx matplotlib -y
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