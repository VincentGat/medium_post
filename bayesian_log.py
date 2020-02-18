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

from pathlib import Path
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

LOG_DIR = Path().absolute() / 'bayes_opt_logs'
LOG_DIR.mkdir(exist_ok=True)

optimizer = BayesianOptimization(optimize_rf, pbounds_forest)
filename = 'log_0.json'
logger = JSONLogger(path=str(LOG_DIR / filename))
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(init_points=10, n_iter=10)

from bayes_opt.util import load_logs

optimizer_load = BayesianOptimization(optimize_rf, pbounds_forest)
all_log = [str(path) for path in LOG_DIR.iterdir()]
load_logs(optimizer_load, logs=all_log)