# Hacktoberfest AIML

Welcome to the **Hacktoberfest AIML**.\
This repo is designed for participants of all skill levels to **learn, contribute, and collaborate** on AI/ML projects.

The tasks are broken down into chunks to suit all levels of difficulty. Tasks in Levels 1/2 are a good starting point if you're getting started with git/GitHub. Once you're familiar with contributing to a repo, you may start by solving active issues in the 'issues' tab. For more experienced folks, you can review the source code, find issues, submit an issue, propose a solution, and try solving it.

If you're participating in Hacktoberfest, please remember that **only accepted (merged) Pull Requests (PRs) count as valid contributions** opening issues alone will not count. All PRs should include a clear explanation of your proposed solution. Reviewers may provide feedback or request changes; please respond to these comments and update your PR as needed to ensure it can be merged smoothly. Constructive discussion is encouraged if you disagree with a review.

---

Once you have forked the repo, then cloned it to local, continue with the following.

## Level 1: Foundations

The aim of this task is to get you started with AIML + GitHub. You will contribute small but essential **utility functions** and **heuristic algorithms**. These functions will later be used in more complex scenarios down the line.

You will have to complete the following:

1. Implement sigmoid, ReLU, tanh functions (using python's math library)
2. Write the logic for vector-dot product / cosine similarity
3. Implement normalization (L1, L2, min-max)
4. Add a simple heuristic function (eg: greedy move for tic-tac-toe)

Directory: `level1-utils/`\
Goal: Learn the repo structure, make your first PR, and get comfortable with GitHub.

---

## Level 2: Models From Scratch

Assuming you have some utility function (say from `level1-utils`, or some other script of yours), it's time to assemble them into models.

Here you will work with building some basic ML and traditional AI models.

For this level, you shall complete the following:

Basic ML algorithms:

1. Implement Linear Regression (with gradient descent)
2. Implement a Perceptron for binary classification
3. Compare models (from 1 and 2) on the dataset in `datasets/binary_classification.csv`
4. Log the following metrics: Accuracy, Time to convergence, Time per prediction

Once complete, create a new file (called `analysis`) with your results and try to explain/theorise the outputs.

(Optional) Traditional AI algorithm: Try implementing the **minimax algorithm** for tic-tac-toe. If you wish to implement other AI algorithms, feel free to do so.

Directory: `level2-models/`

---

## Level 3: Working on real issues

Welcome to the real Open Source Software world. Here on, you will start working on fixing bugs and missing pieces/functionality in the main repo. The main repo has a lot of buggy code that make 'users' frustrated, we have identified some causes of these bugs and reported them in 'issues'. You may pick up on an active issue (this is not assigned to someone) and start working on it. Before you start, please to leave a message under issues to indicate your interest in solving said issue, you may start working on the issue(s) once the repo's maintainer assigns you to it.

Few tags have been created for these issues, like: `level3`, `general`, `critical`, `feature`, `bug`, `contributor-opened`. You may work on any issue that is not `contributor-opened`, these issues have been drafted by another contributor (like you), and they may be working on a fix. Feel free to ask them if they are working on it and act accordingly.

Examples for `level3` issues are:

- Normalization performs L1 norm instead of L2 norm for cosine similarity
- Linear regression model has incorrect matrix shape-initialization
- Accuracy metrics seem to be too low
- README has typos and other *gramer* mistakes
- Shape mismatch in a forward pass (`matmul` dimension error)
- Tensor dtype mismatch (e.g., `float32` vs `long`)
- Device error: CPU vs GPU tensors not aligned
- Incorrect loss reduction (`mean` vs `sum`)
- Batch dimension missing in training loop

Bugs can vary from domain-to-domain. There will be many bugs in AI, ML, DL, etc. You are expected to understand the source of the bug and fix it. Once you are done, when submitting a PR for the fix, mention the approach you took to fix the bug (or add new feature). Please refrain from the use of GPT-like tools, if you do use them, mention it and inform which portion of the contribution was influenced by such tools.

Some typos in the `READMe` are intentionally left for you to discover and fx.

Directory: `src/`\
Goal: Learn to debug, contribute meaningful fixes, and understand common AIML pitfalls.

---

## Level 4: Advanced Extensions

If you're here, it's safe to assume that you understand most of AIML concepts and OSS philosophy. As a result, we will leave you off the hook. You are free to go through the source code, open issues (and tag them with `contributor-opened`), and start working on them.

You are expected to add new features, work on solving logical bugs, etc. You're encouraged to think like a maintainer of the repo: propose new issues, explain their purpose/impact, and solve them.

Some examples include:

- Add new ML models (SVM, Logistic Regression, Decision Tree)
- Add deep learning models (2-layer MLP, CNN, etc)
- Implement modern optimizers (Adam, RMSProp)
- Add SHAP or feature importance explainability
- Improve adversarial AI (Alpha-Beta pruning, A* search)
- Create a visualization dashboard (Streamlit, matplotlib)
- Add CI/CD workflow (tests, linting)

Directory: `src/`\
Goal: Build real-world OSS skills by proposing + solving impactful issues.

---

## Contribution Guide

1. Fork the repo
2. Pick a level and task/issue to work on
3. Clone the repo to local
4. Make your changes
   1. For levels 1/2, make your own dir: `github-username/` and place your code in that dir ONLY
   2. For levels 3/4, you may go to `src/`
5. Commit your changes
   1. Name your commit so that it's meaningful and PG-13
   2. Add a description to the commit if you think its needed
6. Open a PR with a clear description of what you did, how, etc. You must credit all sources that you have used to make said changes.
7. Wait till merge or review
