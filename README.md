# Crypto direction prediction (WIP)

This project aims to develop a novel self-attending LSTM to predict crypto prediciton.

## Table of Contents
1. [AttnLSTM](#AttnLSTM)
2. [Data](#Dataset)

---
# AttnLSTM
## Self-Attention

Self-Attention is a mechanism in neural networks that helps the model focus on different parts of an input sequence when making predictions for each element in the sequence. It's particularly effective in natural language processing, where it helps capture long-term dependencies better than traditional recurrent models.

### Key Components
- **Query (Q)**: The element to which attention is applied.
- **Key (K)**: The element from which the importance of attention is calculated.
- **Value (V)**: The actual content that the model uses in its output.

The attention score for each pair of input elements is calculated as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V
\]

where \( d_k \) is the dimensionality of the query/key vectors, and the softmax function ensures that attention scores are normalized.

In this project, self-attention helps capture complex relationships between tokens, especially for longer sequences, making it a powerful alternative to LSTM.

---

## LSTM (Long Short-Term Memory)

LSTMs are a type of recurrent neural network (RNN) that excel at learning sequential dependencies. Unlike traditional RNNs, they use a memory cell and gating mechanisms to maintain information over long time spans, which reduces issues like the **vanishing gradient problem**.

### Key Components
1. **Cell State (C)**: A memory that stores information over time.
2. **Input Gate**: Controls how much new information is added to the cell state.
3. **Forget Gate**: Controls how much old information is removed from the cell state.
4. **Output Gate**: Determines the final output from the cell state.

The LSTM equations are as follows:

\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]
\[
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]
\[
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
\]
\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]
\[
h_t = o_t * \tanh(C_t)
\]

where:
- \( f_t \): Forget gate
- \( i_t \): Input gate
- \( C_t \): Cell state
- \( o_t \): Output gate
- \( h_t \): Hidden state

In this project, the LSTM model captures temporal dependencies in the sequence, which is useful when each data point in `train.csv` and `test.csv` represents a sequential step.

---

## Dataset 

The dataset consists of two files:
- **`train.csv`**: Used for training the model.
- **`test.csv`**: Used for evaluating model performance.

These files are not uploaded due to their large file size. The files are from this kaggle challenge:[ https://www.kaggle.com/competitions/directional-forecasting-in-cryptocurrencies/data](url)
