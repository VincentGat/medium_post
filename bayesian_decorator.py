from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from pathlib import Path


LOG_DIR = Path().absolute() / 'bayes_opt_logs'
LOG_DIR.mkdir(exist_ok=True)


def optimize_bayes_param(pbounds, init_points, n_iter, X, y, acq='ucb',
                            kappa=2.576, *args_eval, log_dir, **kwargs_eval):
    def optimize_bayes_wo_param(parse_model_param):
        def crossval(*args_model, **kwargs_model):
            estimator = parse_model_param(*args_model, **kwargs_model)
            return cross_val_score(estimator, X=X, y=y, *args_eval, **kwargs_eval).mean()

        optimizer = BayesianOptimization(crossval, pbounds=pbounds)
        optimizer_log_dir = (LOG_DIR / log_dir)
        if optimizer_log_dir.exists():
            all_log = [str(path) for path in optimizer_log_dir.iterdir()]
            load_logs(optimizer, logs=all_log)
            filename = 'log_{}.json'.format(len(all_log))
        else:
            optimizer_log_dir.mkdir()
            filename = 'log_0.json'
        logger = JSONLogger(path=str(optimizer_log_dir / filename))
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

        optimizer.maximize(init_points, n_iter, kappa=kappa, acq=acq)
        best_model = parse_model_param(**optimizer.max['params'])
        best_model.fit(X=X, y=y)
        return best_model

    return optimize_bayes_wo_param


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification


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

    pbounds_forest = {
        'n_estimators': (10, 1000),
        "min_samples_split": (2, 50),
    }


    @optimize_bayes_param(init_points=10, n_iter=10, X=x, y=y
        , pbounds=pbounds_forest, scoring='neg_log_loss', cv=10, log_dir='forest_bayes_trt', kappa=5)
    def forest_bayes(n_estimators, min_samples_split):
        return RandomForestClassifier(n_estimators=int(n_estimators), min_samples_split=int(min_samples_split),
                                      n_jobs=-1)