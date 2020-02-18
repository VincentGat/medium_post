from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score


def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets


x, y = get_data()


def optimize_rf(n_estimators, min_samples_split):
    model = RandomForestClassifier(n_estimators=int(n_estimators), min_samples_split=int(min_samples_split), n_jobs=-1)
    return cross_val_score(model, X=x, y=y, scoring='neg_log_loss', cv=10).mean()


pbounds_forest = {
    'n_estimators': (10, 1000),
    "min_samples_split": (2, 50),
}

optimizer = BayesianOptimization(optimize_rf, pbounds_forest)
optimizer.maximize(init_points=10, n_iter=10)

print(optimizer.max)
# {'target': -0.3807169604445343, 'params': {'min_samples_split': 2.007942621689816, 'n_estimators': 909.1394533822921}}
