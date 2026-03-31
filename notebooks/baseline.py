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
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

    return (
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        TfidfVectorizer,
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        pd,
        plt,
        precision_score,
        re,
        recall_score,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(pd):
    #разделяем датасет на метку и признаки
    data=pd.read_csv('data\processed\clean_data.csv', delimiter=',')
    data = data.dropna(subset=['text']) #мы сохранили чистый датасет на прошлом этапе, но при read csv появляются пустые ячейки (надо фиксить, но пока просто удаляем, потому что количество пустых строк в столбце text менее 1 процента)
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
    X_train, X_test, y_train_logreg, y_test_logreg = train_test_split(
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
    return X_test, X_train, preprocessor, y_test_logreg, y_train_logreg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    baseline - логистическая регрессия (хорошо интерпретируема, хороша работает с текст данными и отправная точка в работе с NLP задачами)
    """)
    return


@app.cell
def _(LogisticRegression, Pipeline, X_train, preprocessor, y_train_logreg):
    model=Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000, #количество итераций
            random_state=17
        ))
    ])
    model.fit(X_train,y_train_logreg)
    return (model,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    model,
    pd,
    y_test_logreg,
    y_train_logreg,
):
    y_train_prediction_logreg=model.predict(X_train)
    y_test_prediction_logreg=model.predict(X_test)
    df=pd.DataFrame(
        {"Какая выборка?": ["Train","Test"],
         "Accuracy метрика": [accuracy_score(y_train_logreg, y_train_prediction_logreg),accuracy_score(y_test_logreg,y_test_prediction_logreg)],
         "F1-мера": [f1_score(y_train_logreg, y_train_prediction_logreg),f1_score(y_test_logreg,y_test_prediction_logreg)]
            }
    )
    print(df)
    return y_test_prediction_logreg, y_train_prediction_logreg


@app.cell
def _(auc, plt, roc_curve, y_test_logreg, y_test_prediction_logreg):
    #ROC-AUC
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test_logreg, y_test_prediction_logreg) #false positive rate (доля ложноположительных срабатываний), true positive rate (чувствительность recall)
    roc_auc_logreg = auc(fpr_logreg, tpr_logreg) #AUC = 1 — идеальная классификация, 0.5 — случайная

    plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.3f})')

    plt.plot([0,1],[0,1],'--') ## Эта линия соответствует случайной модели (AUC=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('image/logreg_roc_curve.png')
    plt.show()
    return


@app.cell
def _(
    confusion_matrix,
    pd,
    precision_score,
    recall_score,
    y_test_logreg,
    y_test_prediction_logreg,
    y_train_logreg,
    y_train_prediction_logreg,
):
    cm = confusion_matrix(y_test_logreg, y_test_prediction_logreg) #матрица неточностей, показывающая TP, FN, TN, FP
    cm_df = pd.DataFrame(
        cm,
        index=['Negative', 'Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    print(cm_df)
    print(f"Precission for train: {precision_score(y_train_logreg, y_train_prediction_logreg):.3f}\nRecall for train: {recall_score(y_train_logreg, y_train_prediction_logreg):.3f}\nPrecission for test: {precision_score(y_test_logreg, y_test_prediction_logreg):.3f}\nRecall for test: {recall_score(y_test_logreg, y_test_prediction_logreg):.3f}")
    return


@app.cell
def _(model, re):
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coef = model.named_steps['classifier'].coef_[0]

    top_pos = coef.argsort()[-5:]
    top_neg = coef.argsort()[:5]

    print("=== POSITIVE WORDS ===")
    for i in top_pos:
        clean_name = re.sub(r'^text__', '', feature_names[i])
        print(f"{clean_name}: {coef[i]:.1f}")

    print("\n=== NEGATIVE WORDS ===")
    for i in top_neg:
        clean_name = re.sub(r'^text__', '', feature_names[i])
        print(f"{clean_name}: {coef[i]:.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Итог
    Собрана рабочая базовая модель логистической регрессии.

    ##Результаты
    |Метрика|Train|Test|
    |:-----:|:----:|:--:|
    |Accuracy|0.795597|0.788302|
    |F1-мера|0.799780|0.792749|
    |Recall|0.817|0.810|
    |Precission|0.784|0.776|

    ##Матрица неточностей
    ||Прогноз y>0|Прогноз y<0|
    |:-----:|:----:|:--:|
    |Реал y>0|129274|30346|
    |Реал y<0|37247|122422|

    , где TP = 129274, TN = 122422, FP = 37247, FN = 30346

    Слова имеющие наибольший вес в модели:

    |Позитивные слова: вес|Негативные слова: вес|
    |:-----:|:----:|
    |thank: 4.5|sad: -11.2|
    |dont sad: 4.5|poor: -6.6|
    |thanks: 4.7|bummed: -6.6|
    |isnt bad: 5.0|miss: -6.5|
    |wish luck: 5.7|sadly: -6.4|

    ##Следующий шаг - улучшение модели с использованием методов бустинга
    """)
    return


if __name__ == "__main__":
    app.run()
