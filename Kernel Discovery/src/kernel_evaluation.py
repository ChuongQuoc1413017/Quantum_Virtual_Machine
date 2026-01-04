import numpy as np
from typing import Dict
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

@dataclass
class KernelEvaluation:
    model_name: str
    encoding_name: str
    train_accuracy: float
    test_accuracy: float
    train_std: float = 0.0
    test_std: float = 0.0

class QSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.model = None
    
    def fit(self, K_train, y_train):
        self.model = SVC(kernel="precomputed", C=self.C)
        self.model.fit(K_train, y_train)
        return self
    
    def predict(self, K):
        return self.model.predict(K)
    
    def score(self, K, y):
        return accuracy_score(y, self.predict(K))


MODEL_REGISTER = {
    "SVM": QSVM
}

def evaluate_kernel(K_train, K_test,
                    y_train, y_test,
                    encoding_name, model_name="SVM"):
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


def evaluate_all_models(K_train, K_test,
                        y_train, y_test,
                        encoding_name):
    results = []
    for model_name in MODEL_REGISTER.keys():
        result = evaluate_kernel(
            K_train, K_test, y_train, y_test,
            encoding_name, model_name)
        results.append(result)
    
    return results


def evaluate_all_kernels(kernels, y_train, y_test):
    all_results = {}
    for encoding_name, (K_train, K_test) in kernels.items():
        results = evaluate_all_models(
            K_train, K_test, y_train, y_test, encoding_name)
        all_results[encoding_name] = results
    
    return all_results


def print_summary(all_results: Dict):
    print("SUMMARY")
    print("-" * 60)
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
    if best_config:
        print(f"Best: {best_config[0]} + {best_config[1]} = {best_test_acc:.4f}")
    else:
        print("No results to evaluate")