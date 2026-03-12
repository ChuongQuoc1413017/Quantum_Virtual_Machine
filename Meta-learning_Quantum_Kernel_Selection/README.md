## Project Structure
```
Meta-learning Quantum Kernel Selection/
│
├── Qsun/                          
│   ├── Qcircuit.py
│   ├── Qdata.py
│   ├── Qencodes.py
│   ├── Qgates.py
│   ├── Qkernels.py
│   ├── Qmeas.py
│   └── Qwave.py
│
├── datasets/
│   ├── load_data.py             
│   └── *.csv                     
│
├── results/                        
│   ├── *.csv                                          
│   └── plots/                     
│
├── src/
│   ├── config.py                  
│   └── kernel_evaluation.py       
│
├── [1] Quantum Learning.ipynb     
├── [2] Majority Voting.ipynb      
├── [2] LOOCV.ipynb               
```

## Execution Order

1. **`Quantum Learning.ipynb`** — Run this first. Loads all datasets, computes quantum kernel matrices (9 ansätze × 3 ML models × 10 runs), extracts 24 complexity metrics, and generates the Synthetic Training Dataset. Outputs are saved to `results/`.

2. **`Majority Voting.ipynb`** and **`LOOCV.ipynb`** — Run after Step 1, in any order (they are independent). Both read the CSV files from `results/`, train recommendation models, evaluate accuracy across Task-A and Task-B (single metric vs all metrics), and perform inference on 7 new test datasets.
