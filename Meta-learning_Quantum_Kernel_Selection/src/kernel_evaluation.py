import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel as GPKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from src.config import *
from Qsun.Qkernels import state_product

def kernel_matrix(X_train: np.ndarray, 
                  X_test: np.ndarray,
                  encoding_name: str, 
                  n_layers: int = 2,
                  params: Optional[np.ndarray] = None, 
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Compute quantum kernel matrices using squared fidelity between encoded quantum states."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    encoded_train = []
    for i in range(n_train):
        state = encode_sample(X_train[i], encoding_name, n_layers, params)
        encoded_train.append(state)
    
    encoded_test = []
    for i in range(n_test):
        state = encode_sample(X_test[i], encoding_name, n_layers, params)
        encoded_test.append(state)
    
    K_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i, n_train):
            k_ij = state_product(encoded_train[i], encoded_train[j]) ** 2
            K_train[i, j] = k_ij
            K_train[j, i] = k_ij
    
    K_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = state_product(encoded_test[i], encoded_train[j]) ** 2
    
    return K_train, K_test

def total_kernels(X_train: np.ndarray,
                  X_test: np.ndarray,
                  encoding_names: Optional[List[str]] = None, 
                  n_layers: int = 2,
                  random_state: int = 42,
                  verbose: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute kernel matrices for all specified quantum encoding methods."""
    if encoding_names is None:
        encoding_names = get_available_encodings()
    
    results = {}

    for name in encoding_names:
        try:
            if verbose:
                print(f"Computing {name}...", end=' ')
            
            K_train, K_test = kernel_matrix(
                X_train, X_test, name, n_layers, 
                random_state=random_state
            )
            
            results[name] = (K_train, K_test)
            
            if verbose:
                print(f"✓")
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
    
    return results


def get_available_encodings() -> List[str]:
    return list(ENCODING_REGISTER.keys())

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

class PrecomputedKernel(GPKernel):

    def __init__(self, K_train: np.ndarray):
        self.K_train = K_train
        self.n_train = K_train.shape[0]
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        
        X_idx = X.flatten().astype(int)
        Y_idx = Y.flatten().astype(int)
        
        K = self.K_train[np.ix_(X_idx, Y_idx)]
        
        if eval_gradient:
            return K, np.zeros((X.shape[0], Y.shape[0], 0))
        
        return K
    
    def diag(self, X):
        X_idx = X.flatten().astype(int)
        return np.diag(self.K_train)[X_idx]  # predictive variance
    
    def is_stationary(self):
        return False
    
    def clone_with_theta(self, theta):
        return PrecomputedKernel(self.K_train)
    
    @property
    def theta(self):
        return np.array([]) # no hyperparameter
    
    @theta.setter
    def theta(self, value):
        pass
    
    @property
    def bounds(self):
        return np.array([]).reshape(0, 2) # neglect optimization
    
    def set_matrix(self, new_matrix: np.ndarray):
        self.K_train = new_matrix
        self.n_train = new_matrix.shape[0]


class QGPC:
    def __init__(self, max_iter_predict: int = 100):
        self.max_iter_predict = max_iter_predict
        self.model = None
        self.kernel = None
        self.K_train = None
        self.y_train = None
        self.n_train = None
    
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        self.K_train = K_train
        self.y_train = y_train
        self.n_train = K_train.shape[0]
        
        K_reg = K_train + 1e-6 * np.eye(K_train.shape[0]) # regularization for positive definite
        
        self.kernel = PrecomputedKernel(K_reg)
        
        self.model = GaussianProcessClassifier(
            kernel=self.kernel,
            max_iter_predict=self.max_iter_predict,
            n_restarts_optimizer=0
        )
        
        X_indices = np.arange(self.n_train).reshape(-1, 1) # use indices instead of raw data
        self.model.fit(X_indices, y_train)
        return self
    
    def predict(self, K_test: np.ndarray) -> np.ndarray:
        n_test = K_test.shape[0]
        
        K_extended = np.block([
            [self.K_train, K_test.T],
            [K_test, np.eye(n_test)]
        ])
        
        K_reg = K_extended + 1e-6 * np.eye(K_extended.shape[0])
        
        if self.model.n_classes_ == 2:
            self.model.base_estimator_.kernel_.set_matrix(K_reg)
        else:
            for estimator in self.model.base_estimator_.estimators_:
                estimator.kernel_.set_matrix(K_reg)
        
        X_test_indices = np.arange(self.n_train, self.n_train + n_test).reshape(-1, 1)
        return self.model.predict(X_test_indices)
    
    def score(self, K: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(K))

class QKRC:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
        self.classes_ = None
        self.is_binary = None
    
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        self.classes_ = np.unique(y_train)
        self.is_binary = len(self.classes_) == 2
        
        self.model = KernelRidge(alpha=self.alpha, kernel="precomputed")
        
        if self.is_binary:
            y_binary = np.where(y_train == self.classes_[0], -1, 1)
            self.model.fit(K_train, y_binary)
        else:
            n_classes = len(self.classes_)
            Y_onehot = np.zeros((len(y_train), n_classes))
            for i, c in enumerate(self.classes_):
                Y_onehot[y_train == c, i] = 1
            self.model.fit(K_train, Y_onehot)
        
        return self
    
    def predict(self, K: np.ndarray) -> np.ndarray:
        pred_raw = self.model.predict(K)
        
        if self.is_binary:
            pred_binary = np.where(pred_raw >= 0, 1, -1)
            return np.where(pred_binary == -1, self.classes_[0], self.classes_[1])
        else:
            pred_idx = np.argmax(pred_raw, axis=1)
            return self.classes_[pred_idx]
    
    def score(self, K: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(K))

MODEL_REGISTER = {
    "SVM": QSVM,
    "GPC": QGPC,
    "KRC": QKRC,
}

def evaluate_kernel(K_train: np.ndarray, 
                   K_test: np.ndarray,
                   y_train: np.ndarray, 
                   y_test: np.ndarray,
                   encoding_name: str, 
                   model_name: str = "SVM",
                   **model_kwargs) -> KernelEvaluation:
    if model_name not in MODEL_REGISTER:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTER.keys())}"
        )
    
    model_class = MODEL_REGISTER[model_name]
    model = model_class(**model_kwargs)
    
    try:
        model.fit(K_train, y_train)
        train_acc = model.score(K_train, y_train)
        test_acc = model.score(K_test, y_test)
    except Exception as e:
        print(f"Warning: {model_name} failed for {encoding_name}: {e}")
        train_acc = 0.0
        test_acc = 0.0
    
    return KernelEvaluation(
        model_name=model_name,
        encoding_name=encoding_name,
        train_accuracy=train_acc,
        test_accuracy=test_acc
    )

def evaluate_all_models(K_train: np.ndarray,
                       K_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray,
                       encoding_name: str) -> List[KernelEvaluation]:

    results = []
    for model_name in MODEL_REGISTER.keys():
        result = evaluate_kernel(
            K_train, K_test, y_train, y_test,
            encoding_name, model_name
        )
        results.append(result)
    return results

def evaluate_all_kernels(kernels: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         y_train: np.ndarray,
                         y_test: np.ndarray) -> Dict[str, List[KernelEvaluation]]:

    all_results = {}
    for encoding_name, (K_train, K_test) in kernels.items():
        results = evaluate_all_models(
            K_train, K_test, y_train, y_test, encoding_name
        )
        all_results[encoding_name] = results
    return all_results


def evaluate_kernel_multiple_runs(K_trains: List[np.ndarray],
                                  K_tests: List[np.ndarray],
                                  y_trains: List[np.ndarray],
                                  y_tests: List[np.ndarray],
                                  encoding_name: str,
                                  model_name: str = "SVM") -> KernelEvaluation:

    train_accs = []
    test_accs = []
    
    for K_tr, K_te, y_tr, y_te in zip(K_trains, K_tests, y_trains, y_tests):
        result = evaluate_kernel(K_tr, K_te, y_tr, y_te, encoding_name, model_name)
        train_accs.append(result.train_accuracy)
        test_accs.append(result.test_accuracy)
    
    return KernelEvaluation(
        model_name=model_name,
        encoding_name=encoding_name,
        train_accuracy=np.mean(train_accs),
        test_accuracy=np.mean(test_accs),
        train_std=np.std(train_accs),
        test_std=np.std(test_accs)
    )

def summary(all_results: Dict[str, List[KernelEvaluation]]):
    print("\n" + "=" * 70)
    print("KERNEL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Encoding':<22} {'Model':<6} {'Train':<18} {'Test':<18}")
    print("-" * 70)
    
    best_test_acc = 0
    best_config = None
    
    for encoding_name, results in all_results.items():
        for r in results:
            train_str = f"{r.train_accuracy:.4f}"
            test_str = f"{r.test_accuracy:.4f}"
            
            if r.train_std > 0:
                train_str += f" ± {r.train_std:.4f}"
                test_str += f" ± {r.test_std:.4f}"
            
            print(f"{r.encoding_name:<22} {r.model_name:<6} {train_str:<18} {test_str:<18}")
            
            if r.test_accuracy > best_test_acc:
                best_test_acc = r.test_accuracy
                best_config = (r.encoding_name, r.model_name)
    
    print("-" * 70)
    if best_config:
        print(f"Best: {best_config[0]} + {best_config[1]} = {best_test_acc:.4f}")
    print("=" * 70)


def best_kernel_per_model(all_results: Dict[str, List[KernelEvaluation]]) -> Dict[str, Tuple[str, float]]:
    best_per_model = {}
    
    for model_name in MODEL_REGISTER.keys():
        best_acc = 0
        best_kernel = None
        
        for encoding_name, results in all_results.items():
            for r in results:
                if r.model_name == model_name and r.test_accuracy > best_acc:
                    best_acc = r.test_accuracy
                    best_kernel = encoding_name
        
        if best_kernel:
            best_per_model[model_name] = (best_kernel, best_acc)
    
    return best_per_model


def get_consensus_best_kernel(all_results: Dict[str, List[KernelEvaluation]]) -> str:
    """Find the quantum kernel with highest average performance across all models."""   
    kernel_scores = {}
    
    for encoding_name, results in all_results.items():
        accs = [r.test_accuracy for r in results]
        kernel_scores[encoding_name] = np.mean(accs)
    
    return max(kernel_scores, key=kernel_scores.get)

def evaluation(X_train: np.ndarray,
                  X_test: np.ndarray,
                  y_train: np.ndarray,
                  y_test: np.ndarray,
                  encoding_names: Optional[List[str]] = None,
                  n_layers: int = 2,
                  random_state: int = 42) -> Dict[str, List[KernelEvaluation]]:

    print("Computing quantum kernels...")
    kernels = total_kernels(
        X_train, X_test,
        encoding_names=encoding_names,
        n_layers=n_layers,
        random_state=random_state,
        verbose=True
    )
    
    print("\nEvaluating all models...")
    results = evaluate_all_kernels(kernels, y_train, y_test)
    
    summary(results)
    
    return results