import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedGroupKFold:
    """5-fold splitter that keeps patient groups intact while stratifying by class."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        groups = np.array(groups)
        y      = np.array(y)
        unique_groups = np.unique(groups)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_groups)

        group_label = {
            g: np.argmax(np.bincount(y[groups == g]))
            for g in unique_groups
        }
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                               random_state=self.random_state)
        for tr_g, te_g in skf.split(unique_groups,
                                     [group_label[g] for g in unique_groups]):
            train_groups = unique_groups[tr_g]
            test_groups  = unique_groups[te_g]
            yield (
                np.where(np.isin(groups, train_groups))[0],
                np.where(np.isin(groups, test_groups))[0],
            )
