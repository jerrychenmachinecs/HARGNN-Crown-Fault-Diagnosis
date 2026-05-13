# HARGNN-Crown-Fault-Diagnosis
Hierarchical Attention Recurrent Graph Neural Network (HARGNN) with Causal Inference for Intelligent Diagnosis of Hot-Rolled Strip Steel Crown Faults. Implementation of the paper "Intelligent diagnosis of hot-rolled strip steel convexity faults based on hierarchical attention cycle graph networks and causal inference
# HARGNN-Crown-Fault-Diagnosis

Official implementation of "Intelligent diagnosis of hot-rolled strip steel convexity faults based on hierarchical attention cycle graph networks and causal inference" published in Applied Soft Computing (2026).

## Overview

This repository contains the implementation of a novel intelligent diagnostic method for hot-rolled strip steel crown faults, combining Hierarchical Attention Recurrent Graph Neural Networks (HARGNN) with causal inference techniques.

## Key Features

- Hierarchical Attention Mechanism (Parameter-level + Stand-level)
- Graph Attention Networks (GAT) for intra-stand parameter interactions
- Gated Recurrent Units (GRU) for cross-stand dynamic evolution
- Cost-sensitive learning for imbalanced data
- Causal inference module based on PC algorithm
- SHAP-based interpretability analysis

## Performance

- Accuracy: 91.23%
- Macro-F1: 0.7612
- Identified 138 significant causal relationships
- Outperforms STGNN, GAT, LSTM, and other baseline methods

## Requirements
torch>=1.13.0
torch-geometric>=2.2.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.41.0
networkx>=2.6.0
scipy>=1.7.0
