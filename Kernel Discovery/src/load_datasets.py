import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from sklearn.datasets import load_iris, make_moons, make_circles, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()

@dataclass
class DatasetInfo:
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: Dict[int, int]
    source: str

class DatasetLoader:
    
    def __init__(self, data_dir: Optional[str] = None, random_state: int = 42):
        self.data_dir = Path(data_dir) if data_dir else SCRIPT_DIR
        self.random_state = random_state
        self._datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._metadata: Dict[str, DatasetInfo] = {}
    
    def _compute_class_distribution(self, y: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique.astype(int), counts.astype(int)))
    
    def _add_dataset(self, name: str, X: np.ndarray, y: np.ndarray, source: str) -> None:
        self._datasets[name] = (X, y)
        self._metadata[name] = DatasetInfo(
            name=name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
            class_distribution=self._compute_class_distribution(y),
            source=source
        )
    
    def load_circle(self, n_samples: int = 100, noise: float = 0.1, 
                    factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_circles(
            n_samples=n_samples, 
            noise=noise, 
            factor=factor,
            random_state=self.random_state
        )
        self._add_dataset("Circle", X, y, "sklearn.make_circles")
        return X, y
    
    def load_blobs(self, n_samples: int = 1000, n_features: int = 2,
                centers: int = 2, cluster_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features, 
            centers=centers,       
            cluster_std=cluster_std,  
            random_state=self.random_state
        )
        name = f"Blobs_F{n_features}C{centers}"
        self._add_dataset(name, X, y, f"sklearn.make_blobs(f={n_features},c={centers})")
        return X, y
    
    def load_moons(self, n_samples: int = 100, 
                   noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_moons(
            n_samples=n_samples, 
            noise=noise,
            random_state=self.random_state
        )
        self._add_dataset("Moons", X, y, "sklearn.make_moons")
        return X, y
    
    def load_iris(self, binary: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X, y = load_iris(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Iris", X, y, "sklearn.load_iris")
        return X, y
    
    def load_pima(self, filename: str = "pima-indians-diabete.csv") -> Tuple[np.ndarray, np.ndarray]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Pima", X, y, str(filepath))
        return X, y
    
    def load_banknote(self, filename: str = "BankNote_Authentication.csv") -> Tuple[np.ndarray, np.ndarray]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Banknote", X, y, str(filepath))
        return X, y
    
    def load_haberman(self, filename: str = "haberman.csv") -> Tuple[np.ndarray, np.ndarray]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        if y.min() == 1:
            y = y - 1
        self._add_dataset("Haberman", X, y, str(filepath))
        return X, y
    
    def load_all(self, verbose: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        
        blobs_configs = [
            (2, 2), (2, 3), (2, 4),  # 2 features
            (4, 2), (4, 3), (4, 4),  # 4 features
        ]
        for n_features, centers in blobs_configs:
            name = f"Blobs_F{n_features}C{centers}"
            try:
                X, y = make_blobs(
                    n_samples=1000,
                    n_features=n_features,
                    centers=centers,
                    cluster_std=0.5,
                    random_state=self.random_state
                )
                self._add_dataset(name, X, y, f"sklearn.make_blobs(f={n_features},c={centers})")
            except Exception as e:
                if verbose:
                    print(f"  Skipped {name}: {e}")

        loaders = [
            ("Circle", self.load_circle),
            ("Moons", self.load_moons),
            ("Iris", self.load_iris),
            ("Pima", self.load_pima),
            ("Banknote", self.load_banknote),
            ("Haberman", self.load_haberman),
        ]

        for name, loader in loaders:
            try:
                if verbose:
                    """print(f"\nLoading {name}...")"""
                X, y = loader()
                if verbose:
                    info = self._metadata[name]
                    """print(f"  {info.n_samples} samples, {info.n_features} features, {info.n_classes} classes")"""
            except FileNotFoundError as e:
                if verbose:
                    print(f"  Skipped: {e}")
                continue
    
        return self._datasets.copy()
    
    def get_metadata(self, name: Optional[str] = None):
        if name:
            return self._metadata.get(name)
        return self._metadata.copy()
    
    @property
    def available_datasets(self) -> List[str]:
        return list(self._datasets.keys())

class DatasetPreprocessor:
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_models: Dict[str, PCA] = {}
        self.pca_info: Dict[str, Dict] = {}
    
    def preprocess(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        test_size: float = 0.2,
        feature_range: Tuple[float, float] = (0, np.pi),
        max_qubits: Optional[int] = None, 
        verbose: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        
        results = {}
        for name, (X, y) in datasets.items():
            original_features = X.shape[1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
            if max_qubits is not None and original_features > max_qubits:
                pca = PCA(n_components=max_qubits, random_state=self.random_state)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                
                self.pca_models[name] = pca
                self.pca_info[name] = {
                    'original_features': original_features,
                    'reduced_features': max_qubits,
                    'explained_variance_ratio': pca.explained_variance_ratio_.sum()
                }
            scaler = MinMaxScaler(feature_range=feature_range)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            results[name] = (X_train, X_test, y_train, y_test)
        
        return results


def load_datasets(
    data_dir: Optional[str] = None,
    test_size: float = 0.20,
    feature_range: Tuple[float, float] = (0, np.pi),
    max_qubits: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    
    loader = DatasetLoader(data_dir=data_dir, random_state=random_state)
    raw_datasets = loader.load_all(verbose=verbose)
    
    preprocessor = DatasetPreprocessor(random_state=random_state)
    return preprocessor.preprocess(
        raw_datasets,
        test_size=test_size,
        feature_range=feature_range,
        max_qubits=max_qubits,
        verbose=verbose
    )

#if __name__ == "__main__":
    datasets = load_datasets(feature_range=(0, np.pi))
    
    print("Verification:")
    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        print(f"  {name}: {X_tr.shape[1]} features, range=[{X_tr.min():.3f}, {X_tr.max():.3f}]")