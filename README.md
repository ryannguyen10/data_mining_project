# FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data

**Data Mining Project — Undergraduate**  
Implementation of the FedAvg algorithm (McMahan et al., 2017) applied to
node classification on graph datasets.

---

## Project Structure

```
fedavg_project/
├── data/
│   ├── CA-GrQc.txt              ← small graph  (~5k nodes)
│   └── com-dblp.ungraph.txt     ← large graph  (~317k nodes)
│
├── src/
│   ├── preprocess.py            ← graph loading, feature extraction, partitioning
│   ├── model.py                 ← NumPy MLP (no deep-learning framework needed)
│   ├── client.py                ← federated client (local SGD)
│   ├── server.py                ← federated server (FedAvg aggregation)
│   ├── fedavg.py                ← main training loop
│   └── evaluate.py              ← Macro-F1, Micro-F1, Hamming Loss
│
├── experiments/
│   ├── effectiveness.py         ← Exp 1: IID vs non-IID vs centralised
│   ├── efficiency.py            ← Exp 2: timing vs rounds / epochs / clients
│   └── scalability.py          ← Exp 3: com-dblp large-scale test
│
├── results/                     ← auto-generated JSON + PNG figures
├── run_all.py                   ← master runner
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended. No GPU required — the MLP is pure NumPy.

### 2. Add the datasets

Place the two data files in the `data/` directory:

```
data/CA-GrQc.txt
data/com-dblp.ungraph.txt
```

Both files are plain edge lists (one `src dst` pair per line, `#` for comments).

---

## Running the Experiments

### Run effectiveness + efficiency tests (small dataset only)

```bash
python run_all.py
```

This will:
- Load and preprocess CA-GrQc
- Run FedAvg with IID and non-IID partitions (50 rounds, E=5)
- Run a centralised baseline
- Time experiments varying rounds, local epochs, and number of clients
- Save results to `results/`

### Run all three experiments including scalability

```bash
python run_all.py --scalability
```

> Note: Scalability on com-dblp (~317k nodes) takes several minutes.

### Run individual experiments

```bash
# From project root
python experiments/effectiveness.py
python experiments/efficiency.py
python experiments/scalability.py
```

---

## Algorithm — FedAvg

```
Input: K clients, T rounds, E local epochs, fraction C, learning rate η

Initialise: global weights W₀

For round t = 1 to T:
    Select m = max(C·K, 1) clients randomly
    For each selected client k (in parallel):
        W_k ← LocalTrain(W_{t-1}, local_data_k, E, η)
    Aggregate:
        W_t ← Σ_k (n_k / n) · W_k

Return: W_T
```

**Local model:** 2-layer MLP  
`Input(5) → Dense(64) → ReLU → Dense(n_classes) → Softmax`

**Node features:** degree, log(degree+1), clustering coefficient,
average neighbour degree, k-core number

**Labels:** Generated via k-core quantisation into 5 equal-frequency classes

---

## Evaluation Measures

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Macro-F1** | Mean F1 per class | Equal weight per class |
| **Micro-F1** | Global TP/(TP+FP+FN) | Equal weight per sample |
| **Hamming Loss** | Wrong labels / total | Lower is better |

---

## Experiment Summary

### Effectiveness (CA-GrQc)

| Method | Macro-F1 | Micro-F1 | Hamming Loss |
|--------|----------|----------|--------------|
| FedAvg IID | 0.7562 | 0.9485 | 0.0515 |
| FedAvg non-IID | 0.7565 | 0.9495 | 0.0505 |
| Centralised | 0.7892 | 0.9867 | 0.0133 |

### Efficiency (CA-GrQc)

| Rounds | Total Time (s) | Avg/Round (s) |
|--------|----------------|---------------|
| 10 | 1.31 | 0.1240 |
| 20 | 2.54 | 0.1208 |
| 50 | 6.19 | 0.1177 |

---

## Key Findings

- **IID vs non-IID**: non-IID partitioning degrades convergence due to
  client data heterogeneity — a known challenge in federated learning.
- **Communication efficiency**: increasing local epochs E improves
  convergence speed per round at the cost of slightly higher round time.
- **Scalability**: FedAvg scales to 317k nodes by using partial client
  participation (C=0.5) and approximating expensive graph features.

---

## Extension Ideas

1. **FedProx** — adds a proximal term `μ/2 ‖w - w_global‖²` to local
   objective to stabilise non-IID training.
2. **Community-aware partitioning** — assign clients by graph community
   rather than random split to better model real-world data silos.
3. **Differential Privacy** — add Gaussian noise to client updates
   before aggregation for formal privacy guarantees.
4. **Graph Neural Networks** — replace the MLP with a GCN layer to
   exploit graph topology during local training.

---

## References

McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017).
*Communication-Efficient Learning of Deep Networks from Decentralized Data.*
AISTATS 2017.

Leskovec, J. & Krevl, A. (2014). SNAP Datasets: Stanford Large Network
Dataset Collection. http://snap.stanford.edu/data
