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
    #Строим baseline модель бинарной классификации
    """)
    return


@app.cell
def _():
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    return (
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        TfidfVectorizer,
        accuracy_score,
        f1_score,
        pd,
        train_test_split,
    )


@app.cell
def _(pd):
    #разделяем датасет на метку и признаки
    data=pd.read_csv('data\processed\clean_data.csv', delimiter=',')
    data = data.dropna(subset=['text'])
    X=data.drop(columns=['y'])
    y=data['y']
    return X, y


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    TfidfVectorizer,
    X,
    train_test_split,
    y,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=17,
        stratify=y
    )
    #применяем tfidf[TF (Term Frequency) — как часто слово встречается в документе
    #IDF (Inverse Document Frequency) — насколько слово редкое во всех документах]
    #к обучающей выборке, чтобы не было утечки данных
    text_trans=TfidfVectorizer(
        ngram_range=(1,3), #учитывает слова отдельные слова, пары и словосочетания до 3 слов
        max_features=20000, #кол-во частотных признаков
        min_df=5, #игнорирует слова, которые встречаются менее чем в 5 твитах
        stop_words='english' #убираем местоимения
    )

    cat_trans=OneHotEncoder(handle_unknown='ignore')
    preprocessor=ColumnTransformer(
        transformers=[
            ('text',text_trans,'text'),
            ('cat', cat_trans, ['day','time'])
        ]
    )

    return X_test, X_train, preprocessor, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    baseline - логистическая регрессия (хорошо интерпретируема, хороша работает с текст данными и отправная точка в работе с NLP задачами)
    """)
    return


@app.cell
def _(LogisticRegression, Pipeline, X_test, X_train, preprocessor, y_train):
    model=Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=17
        ))
    ])
    model.fit(X_train,y_train)
    y_train_prediction=model.predict(X_train)
    y_test_prediction=model.predict(X_test)
    return y_test_prediction, y_train_prediction


@app.cell
def _(
    accuracy_score,
    f1_score,
    pd,
    y_test,
    y_test_prediction,
    y_train,
    y_train_prediction,
):
    pd.DataFrame(
        {"Какая выборка?": ["Выборка train","Выборка test"],
         "Accuracy метрика": [accuracy_score(y_train, y_train_prediction),accuracy_score(y_test,y_test_prediction)],
         "F1-мера": [f1_score(y_train, y_train_prediction),f1_score(y_test,y_test_prediction)]
            }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Итог
    Создан работоспособный пайплайн

    Получены стабильные метрики (~79%)

    Нет переобучения (разница < 1%)

    Следующий этап улучшение показателей путем использования бустинга и более сложных моделей для достижения 90%
    """)
    return


if __name__ == "__main__":
    app.run()
