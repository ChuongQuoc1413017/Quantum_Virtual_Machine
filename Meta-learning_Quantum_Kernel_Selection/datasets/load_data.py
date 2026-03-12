from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import make_moons, make_circles, make_blobs
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
    category: str       # 'synthetic' or 'real-world'
    difficulty: str      # 'easy', 'medium', 'hard'


def make_xor(n_samples=100, noise=0.0, random_state=None):
    rng = np.random.RandomState(random_state)
    n_per_quadrant = n_samples // 4

    X1 = rng.randn(n_per_quadrant, 2) * 0.5 + np.array([1, 1])
    X2 = rng.randn(n_per_quadrant, 2) * 0.5 + np.array([-1, -1])
    X3 = rng.randn(n_per_quadrant, 2) * 0.5 + np.array([1, -1])
    X4 = rng.randn(n_samples - 3 * n_per_quadrant, 2) * 0.5 + np.array([-1, 1])

    X = np.vstack([X1, X2, X3, X4])
    y = np.array([0]*len(X1) + [0]*len(X2) + [1]*len(X3) + [1]*len(X4))

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    return X, y


def make_spiral(n_samples=100, noise=0.0, n_classes=2, random_state=None):
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // n_classes

    X_list, y_list = [], []
    for c in range(n_classes):
        theta = np.linspace(0, 3 * np.pi, n_per_class) + c * (2 * np.pi / n_classes)
        r = np.linspace(0.5, 2, n_per_class)
        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(n_per_class, c))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    return X, y


def make_checkerboard(n_samples=100, grid_size=2, noise=0.0, random_state=None):
    rng = np.random.RandomState(random_state)

    X = rng.rand(n_samples, 2) * grid_size
    y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    return X, y


def make_concentric_rings(n_samples=100, n_rings=3, noise=0.1, random_state=None):
    rng = np.random.RandomState(random_state)
    n_per_ring = n_samples // n_rings

    X_list, y_list = [], []
    for i in range(n_rings):
        radius = (i + 1) * 1.0
        theta = rng.uniform(0, 2 * np.pi, n_per_ring)
        x = radius * np.cos(theta)
        y_coord = radius * np.sin(theta)
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(n_per_ring, i % 2))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    return X, y


class ExtendedDatasetLoader:

    def __init__(self, data_dir=None, random_state=42):
        self.data_dir = Path(data_dir) if data_dir else SCRIPT_DIR
        self.random_state = random_state
        self._datasets = {}
        self._metadata = {}

    def _add_dataset(self, name: str, X: np.ndarray, y: np.ndarray,
                     category: str = 'synthetic', difficulty: str = 'medium'):
        self._datasets[name] = (X, y)
        self._metadata[name] = DatasetInfo(
            name=name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
            category=category,
            difficulty=difficulty
        )


    def load_blobs_easy(self, n_samples=100, n_features=2, centers=2, cluster_std=0.5):
        X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=centers, cluster_std=cluster_std,
                          random_state=self.random_state)
        name = f"Blobs_F{n_features}C{centers}_S{n_samples}"
        self._add_dataset(name, X, y, difficulty='easy')
        return X, y

    def load_blobs_hard(self, n_samples=100, n_features=2, centers=2, cluster_std=2.0):
        X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=centers, cluster_std=cluster_std,
                          random_state=self.random_state)
        name = f"Blobs_hard_F{n_features}C{centers}_std{cluster_std}_S{n_samples}"
        self._add_dataset(name, X, y, difficulty='hard')
        return X, y

    def load_circle(self, n_samples=100, noise=0.1, factor=0.5):
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor,
                            random_state=self.random_state)
        noise_pct = int(noise * 100)
        difficulty = 'easy' if noise < 0.1 else ('medium' if noise < 0.2 else 'hard')
        name = f"Circle_n{noise_pct}pct_f{int(factor*10)}_S{n_samples}"
        self._add_dataset(name, X, y, difficulty=difficulty)
        return X, y

    def load_moons(self, n_samples=100, noise=0.1):
        X, y = make_moons(n_samples=n_samples, noise=noise,
                          random_state=self.random_state)
        noise_pct = int(noise * 100)
        difficulty = 'easy' if noise < 0.15 else ('medium' if noise < 0.25 else 'hard')
        name = f"Moons_n{noise_pct}pct_S{n_samples}"
        self._add_dataset(name, X, y, difficulty=difficulty)
        return X, y

    def load_xor(self, n_samples=100, noise=0.0):
        X, y = make_xor(n_samples=n_samples, noise=noise,
                        random_state=self.random_state)
        if noise > 0:
            name = f"XOR_n{int(noise*100)}pct_S{n_samples}"
            difficulty = 'hard'
        else:
            name = f"XOR_S{n_samples}"
            difficulty = 'medium'
        self._add_dataset(name, X, y, difficulty=difficulty)
        return X, y

    def load_spiral(self, n_samples=100, noise=0.0, n_classes=2):
        X, y = make_spiral(n_samples=n_samples, noise=noise,
                           n_classes=n_classes, random_state=self.random_state)
        if noise > 0:
            name = f"Spiral_n{int(noise*100)}pct_S{n_samples}"
        else:
            name = f"Spiral_S{n_samples}"
        self._add_dataset(name, X, y, difficulty='hard')
        return X, y

    def load_checkerboard(self, n_samples=100, grid_size=2, noise=0.0):
        X, y = make_checkerboard(n_samples=n_samples, grid_size=grid_size,
                                 noise=noise, random_state=self.random_state)
        if noise > 0:
            name = f"Checkerboard_{grid_size}x{grid_size}_n{int(noise*100)}pct_S{n_samples}"
        else:
            name = f"Checkerboard_{grid_size}x{grid_size}_S{n_samples}"
        self._add_dataset(name, X, y, difficulty='hard')
        return X, y

    def load_rings(self, n_samples=100, n_rings=3, noise=0.1):
        X, y = make_concentric_rings(n_samples=n_samples, n_rings=n_rings,
                                     noise=noise, random_state=self.random_state)
        name = f"Rings_{n_rings}r_n{int(noise*100)}pct_S{n_samples}"
        self._add_dataset(name, X, y, difficulty='hard')
        return X, y

    def load_iris(self, binary=True):
        X, y = load_iris(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Iris", X, y, category='real-world', difficulty='easy')
        return X, y

    def load_wine(self, binary=True):
        X, y = load_wine(return_X_y=True)
        if binary:
            mask = y < 2
            X, y = X[mask], y[mask]
        self._add_dataset("Wine", X, y, category='real-world', difficulty='medium')
        return X, y

    def load_breast_cancer(self):
        X, y = load_breast_cancer(return_X_y=True)
        self._add_dataset("BreastCancer", X, y, category='real-world', difficulty='medium')
        return X, y

    def load_pima(self, filename="pima-indians-diabete.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Pima", X, y, category='real-world', difficulty='hard')
        return X, y

    def load_banknote(self, filename="BankNote_Authentication.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self._add_dataset("Banknote", X, y, category='real-world', difficulty='medium')
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
        self._add_dataset("Haberman", X, y, category='real-world', difficulty='hard')
        return X, y

    def load_all_extended(self, verbose=True):

        sample_sizes = [100, 150, 200]

        if verbose:
            print("=" * 60)
            print("LOADING DATASETS")
            print("=" * 60)

        # ── 1.1 Blobs Easy ──
        loaded, skipped = 0, 0
        blobs_easy_configs = [
            (2, 2), (2, 3), (2, 4),
            (3, 2), (3, 3),
            (4, 2), (4, 3), (4, 4)
        ]
        for n_features, centers in blobs_easy_configs:
            for n_samples in sample_sizes:
                try:
                    self.load_blobs_easy(n_samples=n_samples, n_features=n_features,
                                        centers=centers, cluster_std=0.5)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Blobs Easy      : {loaded} loaded, {skipped} skipped")

        # ── 1.2 Blobs Hard ──
        loaded, skipped = 0, 0
        blobs_hard_configs = [
            (2, 2, 1.5), (2, 2, 2.0),
            (2, 3, 2.0),
            (3, 2, 2.0),
            (4, 2, 2.0), (4, 2, 2.5)
        ]
        for n_features, centers, std in blobs_hard_configs:
            for n_samples in sample_sizes:
                try:
                    self.load_blobs_hard(n_samples=n_samples, n_features=n_features,
                                        centers=centers, cluster_std=std)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Blobs Hard      : {loaded} loaded, {skipped} skipped")

        # ── 1.3 Circles ──
        loaded, skipped = 0, 0
        circle_configs = [
            (0.05, 0.3), (0.05, 0.5),
            (0.10, 0.5), (0.10, 0.7),
            (0.15, 0.5), (0.15, 0.8),
            (0.20, 0.5), (0.20, 0.7)
        ]
        for noise, factor in circle_configs:
            for n_samples in sample_sizes:
                try:
                    self.load_circle(n_samples=n_samples, noise=noise, factor=factor)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Circles         : {loaded} loaded, {skipped} skipped")

        # ── 1.4 Moons ──
        loaded, skipped = 0, 0
        for noise in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            for n_samples in sample_sizes:
                try:
                    self.load_moons(n_samples=n_samples, noise=noise)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Moons           : {loaded} loaded, {skipped} skipped")

        # ── 1.5 Concentric Rings ──
        loaded, skipped = 0, 0
        rings_configs = [
            (3, 0.05), (3, 0.10), (3, 0.20),
            (4, 0.05), (4, 0.10), (4, 0.15),
            (5, 0.10), (5, 0.15)
        ]
        for n_rings, noise in rings_configs:
            for n_samples in sample_sizes:
                try:
                    self.load_rings(n_samples=n_samples, n_rings=n_rings, noise=noise)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Concentric Rings: {loaded} loaded, {skipped} skipped")

        # ── 1.6 XOR ──
        loaded, skipped = 0, 0
        for noise in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            for n_samples in sample_sizes:
                try:
                    self.load_xor(n_samples=n_samples, noise=noise)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  XOR             : {loaded} loaded, {skipped} skipped")

        # ── 1.7 Spiral ──
        loaded, skipped = 0, 0
        for noise in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            for n_samples in sample_sizes:
                try:
                    self.load_spiral(n_samples=n_samples, noise=noise)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Spiral          : {loaded} loaded, {skipped} skipped")

        # ── 1.8 Checkerboard ──
        loaded, skipped = 0, 0
        checkerboard_configs = [
            (2, 0.0), (2, 0.05), (2, 0.10),
            (3, 0.0), (3, 0.05), (3, 0.10),
            (4, 0.0), (4, 0.05),
            (5, 0.0)
        ]
        for grid_size, noise in checkerboard_configs:
            for n_samples in sample_sizes:
                try:
                    self.load_checkerboard(n_samples=n_samples, grid_size=grid_size, noise=noise)
                    loaded += 1
                except Exception as e:
                    skipped += 1
        if verbose:
            print(f"  Checkerboard    : {loaded} loaded, {skipped} skipped")

        # ── 2. Real-World Datasets ──
        if verbose:
            print()
        loaded, skipped = 0, 0
        for loader_name, loader in [("Iris", self.load_iris),
                                     ("Wine", self.load_wine),
                                     ("BreastCancer", self.load_breast_cancer)]:
            try:
                loader()
                loaded += 1
            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"    Skip {loader_name}: {e}")

        csv_loaders = [
            ("Pima", self.load_pima),
            ("Banknote", self.load_banknote),
            ("Haberman", self.load_haberman),
        ]
        for name, loader in csv_loaders:
            try:
                loader()
                loaded += 1
            except FileNotFoundError:
                skipped += 1
                if verbose:
                    print(f"    Skip {name}: file not found")
        if verbose:
            print(f"  Real-World      : {loaded} loaded, {skipped} skipped")

        # ── 2.1 Real-World Subsamples ──
        loaded, skipped = 0, 0
        real_world_subsamples = ['BreastCancer', 'Pima', 'Banknote', 'Haberman', 'Wine', 'Iris']
        for ds_name in real_world_subsamples:
            if ds_name in self._datasets:
                X_full, y_full = self._datasets[ds_name]
                for n_samples in [80, 100, 120, 150]:
                    if len(X_full) >= n_samples:
                        try:
                            X_sub, _, y_sub, _ = train_test_split(
                                X_full, y_full, train_size=n_samples,
                                random_state=self.random_state, stratify=y_full
                            )
                            sub_name = f"{ds_name}_S{n_samples}"
                            self._add_dataset(sub_name, X_sub, y_sub,
                                              category='real-world', difficulty='medium')
                            loaded += 1
                        except Exception as e:
                            skipped += 1
        if verbose:
            print(f"  Subsamples      : {loaded} loaded, {skipped} skipped")

        # ── Summary ──
        if verbose:
            stats = self.get_statistics()
            print(f"\nTotal: {stats['total']} datasets "
                  f"(synthetic: {stats['by_category'].get('synthetic', 0)}, "
                  f"real-world: {stats['by_category'].get('real-world', 0)})")
            print("=" * 60)

        return self._datasets.copy()

    def get_metadata(self, name=None):
        if name:
            return self._metadata.get(name)
        return self._metadata.copy()

    @property
    def available_datasets(self):
        return list(self._datasets.keys())

    def get_statistics(self):
        stats = {
            'total': len(self._datasets),
            'by_category': {},
            'by_difficulty': {},
        }

        for name, info in self._metadata.items():
            cat = info.category
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1

            diff = info.difficulty
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1

        return stats

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

def load_extended_datasets(
        data_dir=None,
        test_size=0.2,
        feature_range=(0, np.pi),
        max_qubit=None,
        random_state=42,
        verbose=True):

    loader = ExtendedDatasetLoader(data_dir=data_dir, random_state=random_state)
    raw_datasets = loader.load_all_extended(verbose=verbose)
    preprocessor = DatasetPreprocessor(random_state=random_state)

    return preprocessor.preprocess(
        raw_datasets,
        test_size=test_size,
        feature_range=feature_range,
        max_qubit=max_qubit,
    )