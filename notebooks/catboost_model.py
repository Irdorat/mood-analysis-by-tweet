import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Улучшение качества модели
    Для анализа полярности фрагментов текста CatBoost имеет ряд преимуществ:
    - может работать с категориальными признаками, в том числе и с текстом, из коробки
    - симметричные деревья — меньше переобучаются на шумных текстовых данных
    - скорость инференса — важна, если будете применять модель к новым твитам
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score
    from scipy.sparse import hstack
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        CatBoostClassifier,
        Pool,
        accuracy_score,
        f1_score,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(pd):
    data=pd.read_csv('data\processed\clean_data.csv', delimiter=',')
    data = data.dropna(subset=['text'])
    X=data.drop(columns=['y'])
    y=data['y']
    data.head()
    return X, y


@app.cell
def _(CatBoostClassifier, Pool, X, train_test_split, y):
    X['text_len'] = X['text'].str.len()
    X['word_count'] = X['text'].str.split().str.len()

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
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=7,
    
        loss_function='Logloss',
        eval_metric='F1',
    
        task_type="GPU",
        devices="0",
    
        auto_class_weights='Balanced',
    
        early_stopping_rounds=150,
        bagging_temperature=1,
        random_strength=2,
        verbose=100
    )

    model.fit(train_pool, eval_set=test_pool)
    return X_test, X_train, model, y_test, y_train


@app.cell
def _(X_test, X_train, accuracy_score, f1_score, model, pd, y_test, y_train):
    y_train_prediction_cb=model.predict(X_train)
    y_test_prediction_cb=model.predict(X_test)
    pd.DataFrame(
        {"Какая выборка?": ["Выборка train","Выборка test"],
         "Accuracy метрика": [accuracy_score(y_train, y_train_prediction_cb),accuracy_score(y_test,y_test_prediction_cb)],
         "F1-мера": [f1_score(y_train, y_train_prediction_cb),f1_score(y_test,y_test_prediction_cb)]
            }
    )
    return


@app.cell
def _(CatBoostClassifier, Pool, X, train_test_split, y):
    #Базовый catboost
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=17,
        stratify=y
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=['day', 'time'],
        text_features=['text']
    )

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=['day', 'time'],
        text_features=['text']
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,

        loss_function='Logloss',

        task_type="GPU",
        devices="0",

        verbose=100
    )

    model.fit(train_pool, eval_set=test_pool)
    return X_test, X_train, model, y_test, y_train


if __name__ == "__main__":
    app.run()
