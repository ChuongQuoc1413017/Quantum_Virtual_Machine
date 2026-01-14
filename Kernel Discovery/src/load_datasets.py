from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_moons, make_circles, make_blobs
)
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
        self._rng = np.random.RandomState(random_state)

    def _add_dataset(self, name, X, y):
        self._datasets[name] = (X, y)
        self._metadata[name] = DatasetInfo(
            name=name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_classes=len(np.unique(y))
        )

    def load_circle(self, n_samples=200, noise=0.1, factor=0.5, name=None):
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=self.random_state
        )
        if name is None:
            name = f"Circle_n{int(noise*100):02d}_f{int(factor*10)}"
        self._add_dataset(name, X, y)
        return X, y

    def load_circle_variants(self):
        configs = [
            {"noise": 0.05, "factor": 0.5, "name": "Circle_n05_f5"},
            {"noise": 0.10, "factor": 0.5, "name": "Circle_n10_f5"},
            {"noise": 0.15, "factor": 0.5, "name": "Circle_n15_f5"},
            {"noise": 0.10, "factor": 0.3, "name": "Circle_n10_f3"},
            {"noise": 0.10, "factor": 0.8, "name": "Circle_n10_f8"},
        ]
        for cfg in configs:
            self.load_circle(
                n_samples=200, 
                noise=cfg["noise"], 
                factor=cfg["factor"], 
                name=cfg["name"]
            )


    def load_moons(self, n_samples=200, noise=0.1, name=None):
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=self.random_state
        )
        if name is None:
            name = f"Moons_n{int(noise*100):02d}"
        self._add_dataset(name, X, y)
        return X, y

    def load_moons_variants(self):
        for noise in [0.05, 0.10, 0.15, 0.25]:
            self.load_moons(n_samples=200, noise=noise)

    def load_blobs(self, n_samples=200, n_features=2, centers=2, 
                   cluster_std=0.5, name=None):
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        if name is None:
            name = f"Blobs_F{n_features}C{centers}"
            if cluster_std != 0.5:
                name += f"_std{int(cluster_std*10)}"
        self._add_dataset(name, X, y)
        return X, y

    def load_blobs_variants(self):
        configs = [
            {"n_features": 2, "centers": 2, "cluster_std": 0.5},
            {"n_features": 2, "centers": 3, "cluster_std": 0.5},
            {"n_features": 2, "centers": 4, "cluster_std": 0.5},
            {"n_features": 4, "centers": 2, "cluster_std": 0.5},
            {"n_features": 4, "centers": 3, "cluster_std": 0.5},
            {"n_features": 4, "centers": 4, "cluster_std": 0.5},
            {"n_features": 2, "centers": 2, "cluster_std": 0.3},
            {"n_features": 2, "centers": 2, "cluster_std": 1.0},
            {"n_features": 4, "centers": 2, "cluster_std": 0.3},
            {"n_features": 4, "centers": 2, "cluster_std": 1.0},
        ]
        for cfg in configs:
            self.load_blobs(**cfg)

    def load_xor(self, n_samples=200, cluster_std=0.5):
        centers = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
        X, y_raw = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        y = np.array([0 if i in [0, 1] else 1 for i in y_raw])
        self._add_dataset("XOR", X, y)
        return X, y

    def load_spiral(self, n_samples=200, noise=0.3):
        n = n_samples // 2
        self._rng = np.random.RandomState(self.random_state)
        
        theta = np.sqrt(self._rng.rand(n)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        r_b = -2 * theta - np.pi

        X_a = np.column_stack([
            r_a * np.cos(theta), 
            r_a * np.sin(theta)
        ]) + self._rng.randn(n, 2) * noise
        
        X_b = np.column_stack([
            r_b * np.cos(theta), 
            r_b * np.sin(theta)
        ]) + self._rng.randn(n, 2) * noise

        X = np.vstack([X_a, X_b])
        y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
        self._add_dataset("Spiral", X, y)
        return X, y

    def load_checkerboard(self, n_samples=200, grid_size=2):
        n_per_cell = n_samples // (grid_size ** 2)
        X_list, y_list = [], []
        self._rng = np.random.RandomState(self.random_state)

        for i in range(grid_size):
            for j in range(grid_size):
                X_cell = self._rng.rand(n_per_cell, 2) + [i, j]
                y_cell = np.full(n_per_cell, (i + j) % 2)
                X_list.append(X_cell)
                y_list.append(y_cell)

        X = np.vstack(X_list)
        y = np.hstack(y_list).astype(int)
        self._add_dataset(f"Checkerboard_{grid_size}x{grid_size}", X, y)
        return X, y

    def load_iris(self, binary=True):
        X, y = load_iris(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Iris", X, y)
        return X, y

    def load_wine(self, binary=True):
        X, y = load_wine(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Wine", X, y)
        return X, y

    def load_breast_cancer(self):
        X, y = load_breast_cancer(return_X_y=True)
        self._add_dataset("BreastCancer", X, y)
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

    def load_all(self, include_variants=True, verbose=True):

        if include_variants:
            self.load_blobs_variants()
            self.load_circle_variants()
            self.load_moons_variants()
            
            try:
                self.load_xor()
            except Exception as e:
                if verbose:
                    print(f"  Skipped XOR: {e}")

            try:
                self.load_spiral()
            except Exception as e:
                if verbose:
                    print(f"  Skipped Spiral: {e}")

            try:
                self.load_checkerboard(grid_size=2)
            except Exception as e:
                if verbose:
                    print(f"  Skipped Checkerboard_2x2: {e}")
        else:
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

            try:
                self.load_circle(name="Circle")
            except Exception as e:
                if verbose:
                    print(f"  Skipped Circle: {e}")

            try:
                self.load_moons(name="Moons")
            except Exception as e:
                if verbose:
                    print(f"  Skipped Moons: {e}")

        real_loaders = [
            ("Iris", self.load_iris),
            ("Wine", self.load_wine),
            ("BreastCancer", self.load_breast_cancer),
            ("Pima", self.load_pima),
            ("Banknote", self.load_banknote),
            ("Haberman", self.load_haberman),
        ]

        for name, loader in real_loaders:
            try:
                loader()
            except FileNotFoundError as e:
                if verbose:
                    print(f"  Skipped {name}: {e}")
            except Exception as e:
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
        self.scalers = {}

    def preprocess(self, datasets, test_size=0.2, 
                   feature_range=(0, np.pi), max_qubit=None):
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
            self.scalers[name] = scaler

            results[name] = (X_train, X_test, y_train, y_test)

        return results


def load_datasets(
        data_dir=None,
        test_size=0.2,
        feature_range=(0, np.pi),
        max_qubit=None,
        random_state=42,
        include_variants=True):
    loader = DatasetLoader(data_dir=data_dir, random_state=random_state)
    raw_datasets = loader.load_all(include_variants=include_variants)

    preprocessor = DatasetPreprocessor(random_state=random_state)
    return preprocessor.preprocess(
        raw_datasets,
        test_size=test_size,
        feature_range=feature_range,
        max_qubit=max_qubit,
    )


def get_dataset_summary(datasets):
    print(f"{'Dataset':<25} {'Train':>8} {'Test':>8} {'Features':>10} {'Classes':>10}")
    print("-" * 65)
    for name, (X_tr, X_te, y_tr, y_te) in datasets.items():
        n_classes = len(np.unique(y_tr))
        print(f"{name:<25} {X_tr.shape[0]:>8} {X_te.shape[0]:>8} {X_tr.shape[1]:>10} {n_classes:>10}")
    print("-" * 65)
    print(f"Total: {len(datasets)} datasets")
