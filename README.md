# PCL: Partition-based Continual Learning for Audio Classification 🎧

## 📌 Overview

Continual audio classification requires models to **learn new sound classes over time** without retraining from scratch, while preserving performance on previously learned classes. This involves balancing:

* **Plasticity** — ability to learn new information
* **Stability** — ability to retain prior knowledge

Most existing continual learning (CL) approaches rely on **single monolithic models**, which limits scalability and adaptability to evolving task distributions.

---

## 🚀 Our Approach: PCL

We propose **PCL (Partition-based Continual Learning)** — a **representation-space framework** that replaces a monolithic model with **multiple lightweight experts**.

### Key Idea

* Use **pretrained audio foundation models** to generate embeddings
* Partition the embedding space via **unsupervised clustering**
* Assign a **specialized expert model** to each latent region

This enables:

* Localized adaptation
* Improved scalability
* Better handling of heterogeneous and evolving data

---

## 🧠 Methodology

1. **Feature Extraction**

   * Use pretrained audio models:

     * CLAP
     * AST
     * Wav2Vec2

2. **Latent Space Partitioning**

   * Perform **unsupervised clustering** on embeddings
   * Identify homogeneous regions in representation space

3. **Expert Assignment**

   * Train **lightweight expert models** per cluster

4. **Continual Learning Setup**

   * Class-incremental learning (no access to past data)
   * Exemplar-free setting

---

## 📊 Results

Evaluated on:

* **ESC-50**
* **UrbanSound8K**

---

## 🧪 Experimental Setup

* Continual Learning setting: **Class-Incremental (Exemplar-Free)**
* Feature backbones:

  * CLAP
  * AST
  * Wav2Vec2
* Evaluation metrics:

  * Accuracy
  * Backward Transfer (BWT)

---

## 📁 Repository Structure

```bash
src/pcl/        # Core PCL framework (partitioning, experts, training)
scripts/        # Training and evaluation scripts
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/PCL.git
```

---

## 📈 Key Contributions

* Introduces a **partition-based alternative** to monolithic continual learning
* Demonstrates effectiveness of **multi-expert architectures in representation space**
* Achieves **significant gains in accuracy and knowledge retention**
* Scalable to **heterogeneous and evolving input distributions**

---

## 🔬 Future Work

* Adaptive partitioning strategies
* Expert growth and pruning mechanisms
* Extension to multimodal and streaming settings


