
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def prepare_data(df, target_column, selected_metabolites=None, test_size=0.2, random_state=42):
    """Подготовка данных для обучения"""
    if selected_metabolites is None:
        selected_metabolites = [col for col in df.columns if col not in ['Название образца', 'Группа', 'Код']]
    
    X = df[selected_metabolites]
    y = df[target_column]
    
    # Разделение на train-val и test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    
    # Балансировка классов
    ros = RandomOverSampler(random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    # Кодирование меток
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return {
        'X_train': X_train, 'y_train': y_train, 'y_train_encoded': y_train_encoded,
        'X_test': X_test, 'y_test': y_test, 'y_test_encoded': y_test_encoded,
        'label_encoder': le
    }
    
    
def get_model_configs():
    """Конфигурации моделей и их гиперпараметров для GridSearch"""
    return {
        'LogisticRegression_L1': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1'],
                'solver': ['liblinear', 'saga']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 0.8, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            }
        }
    }

def train_models(X_train, y_train, cv_splits=5):
    """Обучение моделей с подбором гиперпараметров"""
    model_configs = get_model_configs()
    best_models = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    for name, config in model_configs.items():
        print(f"Training {name}...")
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    
    return best_models


def evaluate_models(models, X_test, y_test):
    """Оценка моделей на тестовом наборе"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {'accuracy': acc, 'f1_score': f1}
        print(f"{name} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return results


def save_models(models, path_prefix='models/'):
    """Сохранение обученных моделей"""
    for name, model in models.items():
        filename = f"{path_prefix}{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        print(f"Model {name} saved to {filename}")
        





    


import numpy as np
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, confusion_matrix

def find_optimal_threshold(y_true, y_probs, method='f1', beta=1, cost_fp=1, cost_fn=1):
    """
    Находит оптимальный порог классификации
    :param y_true: истинные метки
    :param y_probs: предсказанные вероятности класса 1
    :param method: 'f1', 'roc', 'pr' (precision-recall), 'cost'
    :param beta: параметр для Fβ-score (используется при method='pr')
    :param cost_fp: стоимость ложноположительного прогноза
    :param cost_fn: стоимость ложноотрицательного прогноза
    :return: оптимальный порог, значение метрики
    """
    if method == 'f1':
        thresholds = np.linspace(0, 1, 100)
        scores = [f1_score(y_true, y_probs >= t) for t in thresholds]
        optimal_idx = np.argmax(scores)
        return thresholds[optimal_idx], scores[optimal_idx]
    
    elif method == 'roc':
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx], j_scores[optimal_idx]
    
    elif method == 'pr':
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        fbeta_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)
        optimal_idx = np.argmax(fbeta_scores)
        return thresholds[optimal_idx], fbeta_scores[optimal_idx]
    
    elif method == 'cost':
        thresholds = np.linspace(0, 1, 100)
        costs = []
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            costs.append(fp * cost_fp + fn * cost_fn)
        optimal_idx = np.argmin(costs)
        return thresholds[optimal_idx], -costs[optimal_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}")