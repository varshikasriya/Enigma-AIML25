# Level-2: Models From Scratch

Welcome to Level-2\
Now that you’ve built some utility functions in **Level 1**, it’s time to assemble them into actual models.\
In this stage, you will implement and experiment with **basic ML and traditional AI models**.

---

## Tasks

### Basic ML Algorithms

1. Implement **Linear Regression** (using Gradient Descent).
2. Implement a **Perceptron** for binary classification.
3. Compare both models on the both datasets provided in: `datasets`
4. Log the following metrics for both models **and both datasets**:
   - Accuracy
   - Time to convergence
   - Time per prediction
   > You are expected to save these metrics in some format, like: JSON, CSV, etc. This way, you can perform the next step (analysis) very easily.

The goal is to understand how each model performs on **linearly separable** and **non-linearly separable** data.

### Analysis

1. Once complete, create a new file (called `analysis.md` or `analysis.txt`) with your results.
2. Then, try to explain/theorize why one model performed better/worse than the other on both datasets.

This task is designed to build your **critical thinking skills** and strengthen your ability to connect **theory with practice**.\
Don’t just report the numbers, reflect on *why* the results turned out the way they did.

### (Optional) Traditional AI

AI isn’t only about deep learning and LLMs, the field existed long before modern techniques came along.
It’s important to understand and implement **classic AI algorithms**, since many real-world problems don’t require the power (or overhead) of deep learning.

- Implement the **Minimax algorithm** for tic-tac-toe.
- You may also implement other AI algorithms of your choice (heuristics, search, etc).

This task is meant to build your appreciation for the **foundations of AI** and sharpen your **problem-solving mindset**, especially in adversarial or rule-based settings.

---

## Directory

All your contributions for this level go inside: `level2-models/<your-github-username>/`

This way, we can keep track of every contributor's work easily, all in one place.

> Please refrain from modifying the contents of other contributors' directories, and keep things clean.

---

## Goal

- Learn how ML models are structured from scratch
- Compare two foundational algorithms
- Understand evaluation metrics and runtime behavior
- Practice documenting and analyzing results

---

## Contribution Checklist

- [ ] Create a directory with your GitHub username under `level2-models/`
- [ ] Add your implementation files (eg: `linear_regression.py`, `perceptron.py`)
- [ ] Ensure models expose consistent methods:
  - `fit(X, y)`
  - `predict(X)`
  - or similar, just keep it consistent for all your work
- [ ] Write a comparison script (`compare.py`) that:
  - Loads the both datasets from `datasets`
  - Trains both models
  - Logs required metrics
- [ ] Create an `analysis.md` or `analysis.txt` file with your explanations
  - Try adding diagrams/plots
- [ ] (Optional) Add your `minimax.py` or other AI models
- [ ] Run all scripts locally to check for errors
- [ ] Commit your changes with a meaningful message (eg: *"Implemented perceptron and comparison with linear regression"*)
- [ ] Open a Pull Request with a clear description of your contributions
- [ ] Respond to review comments if requested and update your PR
