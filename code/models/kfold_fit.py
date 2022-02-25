import csv
import xgboost as xgb
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression


def fitting(model_name, X_train, y_train, X_test, submission_name, kfold_value=5, **kwargs):
    if model_name == 'lr':
        model = LogisticRegression(**kwargs)
    elif model_name == 'xgb':
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False, **kwargs
        )
    
    clf = Pipeline( steps=[
        ('scaling', StandardScaler()),
        ('pred', model)
    ])

    kf = KFold(kfold_value)
    predictions = np.zeros((X_test.shape[0], kfold_value))
    i = 0

    importances = np.zeros((X_train.shape[1], kfold_value))

    for train_index, valid_index in kf.split(X_train, y_train):
        clf.fit(X_train[train_index], y_train[train_index])
        y_pred_train = clf.predict_proba(X_train[train_index])[:,1]
        y_pred_valid = clf.predict_proba(X_train[valid_index])[:,1]

        predictions[:, i] = clf.predict_proba(X_test)[:,1]
        print("train: " + str(log_loss(y_train[train_index], y_pred_train)))
        print("valid: " + str(log_loss(y_train[valid_index], y_pred_valid)))
        print()

        if model_name == 'lr':
            importances[:, i] = clf['pred'].coef_[0]
        elif model_name == 'xgb':
            importances[:, i] = clf['pred'].feature_importances_
        i += 1

    y_pred = np.mean(predictions, axis=1)
    feat_imp = np.mean(importances, axis=1)
    for i, v in enumerate(feat_imp):
        print(f'Feature {i}, score : {v}')

    # Write predictions to a file
    predictions = zip(range(len(y_pred)), y_pred)
    fname = Path('submissions') / submission_name
    with open(fname, "w", newline='') as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id','predicted'])
        for row in predictions:
            csv_out.writerow(row)
