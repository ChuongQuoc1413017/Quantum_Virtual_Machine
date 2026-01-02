import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score

import pandas as pd

@dataclass
class KernelEvaluation:
    model_name: str
    encoding_name: str
    train_accuracy: float
    test_accuracy: float
    train_std: float = 0.0
    test_std: float = 0.0

class QSVM:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.model = None
    
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        self.model = SVC(kernel="precomputed", C=self.C)
        self.model.fit(K_train, y_train)
        return self
    
    def predict(self, K: np.ndarray) -> np.ndarray:
        return self.model.predict(K)
    
    def score(self, K: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(K))
    
class QKR:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.dual_coef_ = None
        self.classes_ = None
        self.y_train_encoded = None
    
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        n_samples = len(y_train)
        
        self.y_train_encoded = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes_):
            self.y_train_encoded[y_train == cls, i] = 1
        
        K_reg = K_train + self.alpha * np.eye(n_samples)
        self.dual_coef_ = np.linalg.solve(K_reg, self.y_train_encoded)
        
        return self
    
    def predict(self, K_test: np.ndarray) -> np.ndarray:
        scores = K_test @ self.dual_coef_
        return self.classes_[np.argmax(scores, axis=1)]
    
    def score(self, K: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(K))
    
MODEL_REGISTER = {
    "SVM": QSVM,
    "KRC": QKR,
}

def evaluate_kernel(K_train: np.ndarray, K_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           encoding_name: str, model_name: str = "SVM",
                           verbose: bool = True) -> KernelEvaluation:
    
    model_class = MODEL_REGISTER[model_name]
    model = model_class()
    
    model.fit(K_train, y_train)
    
    train_acc = model.score(K_train, y_train)
    test_acc = model.score(K_test, y_test)
    
    return KernelEvaluation(
        model_name=model_name,
        encoding_name=encoding_name,
        train_accuracy=train_acc,
        test_accuracy=test_acc
    )

def evaluate_all_models(K_train: np.ndarray, K_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        encoding_name: str, 
                        verbose: bool = True) -> List[KernelEvaluation]:
    
    results = []
    
    for model_name in MODEL_REGISTER.keys():
        result = evaluate_kernel(
            K_train, K_test, y_train, y_test,
            encoding_name, model_name, verbose
        )
        results.append(result)
    
    return results

def evaluate_all_kernels(kernels: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         y_train: np.ndarray, y_test: np.ndarray,
                         n_runs: int = 10, verbose: bool = True) -> Dict:
    
    all_results = {}
    
    for encoding_name, (K_train, K_test) in kernels.items():
        results = evaluate_all_models(
            K_train, K_test, y_train, y_test, encoding_name, verbose
        )
        all_results[encoding_name] = results
    
    return all_results

def print_summary(all_results: Dict):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Encoding':<25} {'Model':<6} {'Train':<8} {'Test':<8}")
    print("-" * 60)
    
    best_test_acc = 0
    best_config = None
    
    for encoding_name, results in all_results.items():
        for r in results:
            print(f"{r.encoding_name:<25} {r.model_name:<6} {r.train_accuracy:<8.4f} {r.test_accuracy:<8.4f}")
            if r.test_accuracy > best_test_acc:
                best_test_acc = r.test_accuracy
                best_config = (r.encoding_name, r.model_name)
    
    print("-" * 60)
    print(f"Best: {best_config[0]} + {best_config[1]} = {best_test_acc:.4f}")
    print("=" * 60)