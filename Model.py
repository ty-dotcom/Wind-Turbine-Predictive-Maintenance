import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class Classifier:
    """Comparing IsolationForest and XGBoost Classifiers on the same data, choosing the better model."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.xgb: XGBClassifier | None = None
        self.lgb: LGBMClassifier | None = None
        self.if_model : IsolationForest | None = None

        self.df = self.df.dropna(axis=1)

        self.x = df.drop(['is_anomaly'], axis=1)
        self.y = df['is_anomaly']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=0.3,
                                                                                random_state=42, stratify=self.y)
        self._train_isolation_forest_model()
        self._train_xgboost_classifer()
        self._train_lgboost_classifier()
        self.model = self._choose_best_model()
        self.y_pred = None


    def get_name(self) -> str:
        """Class name of the model."""
        return self.model.__class__.__name__

    def _train_isolation_forest_model(self) -> None:
        """Fit the IsolationForest using the dataset's anomaly rate as conta"""
        outlier_fraction = len(self.df[self.df['is_anomaly'] == 1]) / float(len(self.df))
        model = IsolationForest(n_estimators=100, contamination=outlier_fraction, random_state=42)

        model.fit(self.x_train)
        self.if_model = model

    def _train_xgboost_classifer(self) -> None:
        """Fit the XGBoost Classifier on the training data."""
        xgbc_model = XGBClassifier(eval_metric='logloss')
        xgbc_model.fit(self.x_train, self.y_train)

        self.xgb = xgbc_model

    def _train_lgboost_classifier(self) -> None:
        lgb_model = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.01,
            feature_fraction=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        lgb_model.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)])
        self.lgb = lgb_model

    def _evaluate_score(self, model) -> tuple:
        """Calculates the weighted precision, recall and f1 score of the model."""
        prediction = model.predict(self.x_test)
        if isinstance(model, IsolationForest):
            # Maps the given -1 for anomalies to 1 instead.
            prediction = [1 if p == -1 else 0 for p in prediction]

        return precision_recall_fscore_support(self.y_test, prediction, average='weighted')

    def _choose_best_model(self) -> IsolationForest | XGBClassifier | LGBMClassifier:
        """Selects the model with the highest weighted f1 score."""
        if_evaluation = self._evaluate_score(self.if_model)
        xgb_evaluation = self._evaluate_score(self.xgb)
        lgb_evaluation = self._evaluate_score(self.lgb)
        models = [(if_evaluation, self.if_model), (lgb_evaluation, self.lgb), (xgb_evaluation, self.xgb),]
        return max(models, key=lambda item: item[0])[1]

    def predict(self, x: pd.DataFrame) -> list:
        """Predicts the Normal vs Anomaly for the given observations."""
        prediction = self.model.predict(x)
        if isinstance(self.model, IsolationForest):
            prediction = [1 if p == -1 else 0 for p in prediction]
        return prediction

    def change_model(self, name) -> None:
        """Manually switches the current model between IsolationForest, XGBoost, LGBMClassifier."""
        if name == 'XGBClassifier':
            self.model = self.xgb
        elif name == 'IsolationForest':
            self.model = self.if_model
        elif name == "LGBMClassifier":
            self.model = self.lgb
        return