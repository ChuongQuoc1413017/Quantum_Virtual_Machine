from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_iris, make_moons, make_circles, make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).parent.resolve()

@dataclass
class DatasetInfo:
    name: str
    n_samples: int
    n_features: int
    n_classes: int

class DatasetLoader:
    def __init__(self, data_dir=None, random_state=42):
        self.data_dir = Path(data_dir) if data_dir else SCRIPT_DIR
        self.random_state = random_state
        self._datasets = {}
        self._metadata = {}

    def _add_dataset(self, name, X, y):
        self._datasets[name] = (X, y)
        self._metadata[name] = DatasetInfo(
            name=name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
        )

    def load_circle(self, n_samples=100, noise=0.1, factor=0.5):
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=self.random_state
        )
        self._add_dataset("Circle", X, y)
        return X, y
    
    def load_blobs(self, n_samples=1000, n_features=2, centers=2, cluster_std=0.5):
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        name = f"Blobs_F{n_features}C{centers}"
        self._add_dataset(name, X, y)
        return X, y

    def load_moons(self, n_samples=100, noise=0.1):
        X, y = make_moons(
            n_samples=n_samples, 
            noise=noise,
            random_state=self.random_state
        )
        self._add_dataset("Moons", X, y)
        return X, y
    
    def load_iris(self, binary=True):
        X, y = load_iris(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Iris", X, y)
        return X, y
    
    def load_pima(self, filename="pima-indians-diabete.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Pima", X, y)
        return X, y
    
    def load_banknote(self, filename="BankNote_Authentication.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Banknote", X, y)
        return X, y
    
    def load_haberman(self, filename="haberman.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        if y.min() == 1:
            y = y - 1
        self._add_dataset("Haberman", X, y)
        return X, y
    
    def load_all(self, verbose=True): 
        blobs_configs = [
            (2, 2), (2, 3), (2, 4),  
            (4, 2), (4, 3), (4, 4), 
        ]
        for n_features, centers in blobs_configs:
            try:
                self.load_blobs(n_features=n_features, centers=centers)
            except Exception as e:
                if verbose:
                    print(f"  Skipped Blobs_F{n_features}C{centers}: {e}")

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
                loader()
            except FileNotFoundError as e:
                if verbose:
                    print(f"  Skipped {name}: {e}")

        return self._datasets.copy()
        
    def get_metadata(self, name=None):
        if name:
            return self._metadata.get(name)
        return self._metadata.copy()
    
    @property
    def available_datasets(self):
        return list(self._datasets.keys())
    
class DatasetPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca_models = {}
        self.pca_info = {}

    def preprocess(self, datasets, test_size=0.2, feature_range=(0, np.pi), max_qubit=None):
        results = {}
        for name, (X, y) in datasets.items():
            original_features = X.shape[1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
            if max_qubit is not None and original_features > max_qubit:
                pca = PCA(n_components=max_qubit, random_state=self.random_state)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                
                self.pca_models[name] = pca
                self.pca_info[name] = {
                    'original_features': original_features,
                    'reduced_features': max_qubit,
                    'explained_variance_ratio': pca.explained_variance_ratio_.sum()
                }
            scaler = MinMaxScaler(feature_range=feature_range)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            results[name] = (X_train, X_test, y_train, y_test)

        return results  

def load_datasets(
        data_dir=None,
        test_size=0.2,
        feature_range=(0, np.pi),
        max_qubit=None,
        random_state=42):
    loader = DatasetLoader(data_dir=data_dir, random_state=random_state)
    raw_datasets = loader.load_all()
    
    preprocessor = DatasetPreprocessor(random_state=random_state)
    return preprocessor.preprocess(
        raw_datasets,
        test_size=test_size,
        feature_range=feature_range,
        max_qubit=max_qubit, 
    )