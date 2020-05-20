# Bayesian_HMM
This project is the experimental part of my undergraduate thesis.

The project includes the things below:

- The main part is in the `main.py` which can set the hyper-parameters.
- The `basic_algorithm.py` includes NIX posterior estimation (VB algorithm is a single function), forward-backward algorithm and Viterbi algorithm. The skills of calculation refers part of `hmmlearn` (https://github.com/hmmlearn/hmmlearn).
- The `core_model.py` includes the four deterministic algorithms introduced by the paper **"Estimation of Viterbi path in Bayesian hidden Markov models"**.
- The `data_generate.py` includes the generation of initial sequences and TMC model data introduced by the paper **"Triplet Markov Chains in hidden signal restoration"**.

