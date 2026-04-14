import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
    from scipy.sparse import hstack
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import pickle
    import os

    return (
        CatBoostClassifier,
        Pool,
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        mo,
        os,
        pd,
        pickle,
        plt,
        precision_score,
        recall_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(pd):
    data=pd.read_csv('data\processed\clean_data.csv', delimiter=',')
    data = data.dropna(subset=['text'])
    X=data.drop(columns=['y'])
    y=data['y']
    X['text_len'] = X['text'].str.len()
    X['word_count'] = X['text'].str.split().str.len()
    X['unique_word_ratio'] = (
        X['text'].str.split().apply(lambda x: len(set(x)) / (len(x) + 1))
    )
    X.head()
    return X, y


@app.cell
def _(CatBoostClassifier, Pool, X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=17,
        stratify=y
    )

    X_train['text'] = X_train['text'].str[:150]
    X_test['text'] = X_test['text'].str[:150]


    train_pool=Pool(
        X_train,
        y_train,
        cat_features=['day','time'],
        text_features=['text']
    )
    test_pool=Pool(
        X_test,
        y_test,
        cat_features=['day','time'],
        text_features=['text']
    )

    model = CatBoostClassifier(
        iterations=7000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=10,

        loss_function='Logloss',
        eval_metric='F1',

        task_type="GPU",
        devices="0",

        auto_class_weights='Balanced',

        early_stopping_rounds=200,
        bagging_temperature=1,
        random_strength=2,
        verbose=100
    )

    model.fit(train_pool, eval_set=test_pool)
    return X_test, X_train, model, y_test, y_train


@app.cell
def _(X_test, X_train, accuracy_score, f1_score, model, pd, y_test, y_train):
    y_train_prediction_cb=model.predict(X_train)
    y_train_prediction_prob_cb=model.predict_proba(X_train)[:,1]
    y_test_prediction_cb=model.predict(X_test)
    y_test_prediction_prob_cb=model.predict_proba(X_test)[:,1]
    df=pd.DataFrame(
        {"Какая выборка?": ["Train","Test"],
         "Accuracy метрика": [accuracy_score(y_train, y_train_prediction_cb),accuracy_score(y_test,y_test_prediction_cb)],
         "F1-мера": [f1_score(y_train, y_train_prediction_cb),f1_score(y_test,y_test_prediction_cb)]
            }
    )
    print(df)
    return (
        y_test_prediction_cb,
        y_test_prediction_prob_cb,
        y_train_prediction_cb,
    )


@app.cell
def _(auc, plt, roc_curve, y_test, y_test_prediction_prob_cb):
    #ROC-AUC
    fpr_cb, tpr_cb, _ = roc_curve(y_test, y_test_prediction_prob_cb) #false positive rate (доля ложноположительных срабатываний), true positive rate (чувствительность recall)
    roc_auc_cb = auc(fpr_cb, tpr_cb) #AUC = 1 — идеальная классификация, 0.5 — случайная

    plt.plot(fpr_cb, tpr_cb, label=f'Catboost Finetune (AUC = {roc_auc_cb:.3f})')

    plt.plot([0,1],[0,1],'--') ## Эта линия соответствует случайной модели (AUC=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('image/tune_catboost_roc_curve.png')
    plt.show()
    return


@app.cell
def _(
    confusion_matrix,
    pd,
    precision_score,
    recall_score,
    y_test,
    y_test_prediction_cb,
    y_train,
    y_train_prediction_cb,
):
    cm = confusion_matrix(y_test, y_test_prediction_cb) #матрица неточностей, показывающая TP, FN, TN, FP
    cm_df = pd.DataFrame(
        cm,
        index=['Negative', 'Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    print(cm_df)
    print(f"Precission for train: {precision_score(y_train, y_train_prediction_cb):.3f}\nRecall for train: {recall_score(y_train, y_train_prediction_cb):.3f}\nPrecission for test: {precision_score(y_test, y_test_prediction_cb):.3f}\nRecall for test: {recall_score(y_test, y_test_prediction_cb):.3f}")
    return


@app.cell
def _(model):
    #catboost напрямую не дает посмотреть важность слов, но дает оценить важность стобцов
    fi = model.get_feature_importance(prettified=True)
    print(fi.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Итог
    Провели настройку гиперпараметров модели и получили прост в 2% качества модели

    ##Результаты
    |Метрика|Train|Test|
    |:-----:|:----:|:--:|
    |Accuracy|0.843899|0.823605|
    |F1-мера|0.843448|0.823060|
    |Recall|0.841|0.821|
    |Precission|0.846|0.825|

    ##Матрица неточностей
    ||Прогноз y>0|Прогноз y<0|
    |:-----:|:----:|:--:|
    |Реал y>0|130992|28628|
    |Реал y<0|27693|131976|

    , где TP = 130992, TN = 131976, FP = 27693, FN = 28628

    ##Следующий шаг - повысить качество модели используя ансамбль
    """)
    return


@app.cell
def _(model, os, pickle):
    os.makedirs('models', exist_ok=True)
    package = {
        # Модели
        'model': model,    
        # Признаки
        'cat_features': ['day', 'time'],
        'text_features': ['text'],
        'text_truncate': 150,

        # Дополнительные признаки, которые добавлялись
        'engineered_features': ['text_len', 'word_count', 'unique_word_ratio']
    }

    # Сохраняем одним файлом
    with open('models/catboost.pkl', 'wb') as f:
        pickle.dump(package, f)
    return


if __name__ == "__main__":
    app.run()
