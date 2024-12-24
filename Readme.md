# Transformer-based English-Bengali Translation Model  

This repository demonstrates the implementation of a **Transformer model** for translation tasks, specifically for translating from English to Bengali. The project focuses on understanding the workings of the Transformer architecture and coding it from scratch using PyTorch.  

---

## Table of Contents  
- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training Details](#training-details)  
- [Results](#results)  
- [Future Work](#future-work)  
- [Acknowledgments](#acknowledgments)  

---

## Introduction  
Transformers have become the de-facto standard for translation tasks due to their ability to process sequences in parallel and leverage the attention mechanism for fast and efficient translations.  

This project implements a basic Transformer model trained on the **Samantar Dataset**. The model is far from being a standard English-Bengali translator and currently produces nearly random Bengali word sequences for given English input sequences.  

The primary goal of this project is to understand the Transformer architecture and build it from scratch.  
Character level encoding is used over here for simplicity, unlike the Byte Pair Encoding algorithm which is widely used and much better in encoding syntatic and semantic information of tokens.

---

## Dataset  
The model is trained on the **Samantar Dataset**, which contains English-Bengali parallel text.  

- **Dataset Link:** [Samantar Dataset](https://www.kaggle.com/datasets/mathurinache/samanantar)  
- The dataset was preprocessed and tokenized to suit the input format required by the Transformer model.  

---

## Model Architecture  
The Transformer model used in this project has the following specifications:  

- **Layers:** 2  
- **Model Dimension (d_model):** 256  
- **Number of Epochs:** 1  

Key Features:  
- Parallel processing of sequences.  
- Scaled dot-product attention mechanism.  
- Encoder-decoder architecture.  

For details, refer to the [Transformer Architecture](https://arxiv.org/abs/1706.03762) paper by Vaswani et al.  

---

## Training Details  
Due to limited computational resources, we trained a very shallow version of the Transformer model:  

| Parameter       | Value    |  
|-----------------|----------|  
| Layers          | 2        |  
| Model Dimension | 256      |  
| Epochs          | 1        |  

**Expected Performance:**  
When trained with:  
- **Layers:** 6  
- **Model Dimension (d_model):** 512  
- **Epochs:** 7–8  
A decent translation performance can be expected.  

---

## Results  
The current model generates sequences of random Bengali words for a given English input sequence.  

This is due to:  
1. Limited training (shallow architecture and low epochs).  
2. Insufficient computational resources for training a deeper model.  

---

## Future Work  
- Train a deeper Transformer model with **d_model = 512**, **layers = 6**, and **epochs = 7–8**.  
- Explore hyperparameter tuning to improve translation accuracy.  
- Experiment with pre-trained embeddings for better initialization.  
- Use GPUs for faster and more efficient training.  

---

## Acknowledgments  
- The **Samantar Dataset** for providing the parallel English-Bengali text corpus.  
- The PyTorch community for its excellent documentation and resources.  
- The authors of the Transformer architecture paper ([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).  

---

## Disclaimer  
This project is purely educational and aims to demonstrate the working of the Transformer model. The current implementation is not intended for production use.  
