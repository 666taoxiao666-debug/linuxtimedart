"""Synthetic R1-B1 evidence only: no project data or experiment is run."""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def demonstrate():
    history = np.tile(np.arange(8.), 20)
    prefix = np.array([1.2, 2.2])
    queries = np.arange(8.)
    def fit(future):
        # One varying H11 coordinate, other coordinates constant.
        values = np.r_[history, prefix, future]
        x = np.c_[values, np.zeros((len(values), 10))]
        scaler = StandardScaler().fit(x)
        model = KMeans(n_clusters=4, random_state=2024, n_init=10).fit(scaler.transform(x))
        q = np.c_[queries, np.zeros((len(queries), 10))]
        labels = model.predict(scaler.transform(q))
        return scaler.mean_[0], labels
    mean_a, a = fit(np.tile([0., 2., 4., 6.], 50))
    mean_b, b = fit(np.tile([0., 20., 40., 60.], 50))
    # Co-membership is invariant to arbitrary permutations of cluster ids.
    changed = np.argwhere((a[:, None] == a) != (b[:, None] == b))
    assert len(changed), 'Expected dependence on future observations'
    pair = changed[0]
    return dict(synthetic_only=True, no_project_data=True,
                same_calibration_and_issue_prefix=True, seed=2024,
                mean_with_future_a=float(mean_a), mean_with_future_b=float(mean_b),
                query_values=queries.tolist(), groups_a=a.tolist(), groups_b=b.tolist(),
                changed_pair=queries[pair].tolist(),
                conclusion='Changing only future target features changes past-query group membership.')


if __name__ == '__main__':
    print(json.dumps(demonstrate(), indent=2))
