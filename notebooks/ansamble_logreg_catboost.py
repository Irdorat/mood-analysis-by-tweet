import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

    from catboost import CatBoostClassifier, Pool
    import matplotlib.pyplot as plt
    import re
    import pickle
    import os

    return (
        CatBoostClassifier,
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        Pool,
        TfidfVectorizer,
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        mo,
        pd,
        pickle,
        plt,
        precision_score,
        re,
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
def _(
    CatBoostClassifier,
    ColumnTransformer,
    LogisticRegression,
    OneHotEncoder,
    Pipeline,
    Pool,
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

    X_train['text'] = X_train['text'].str[:150]
    X_test['text'] = X_test['text'].str[:150]

    tfidf = TfidfVectorizer(
        ngram_range=(1,5), #учитывает слова отдельные слова, пары и словосочетания до 3 слов
        max_features=15000, #кол-во частотных признаков
        min_df=5, #игнорирует слова, которые встречаются менее чем в 5 твитах
        stop_words='english' #убираем местоимения
    )

    cat_trans=OneHotEncoder(handle_unknown='ignore')

    preprocessor=ColumnTransformer(
        transformers=[
            ('text',tfidf,'text'),
            ('cat', cat_trans, ['day','time'])
        ]
    )

    lr = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=3000,
            random_state=17,
            solver='saga'

        ))
    ])

    lr.fit(X_train, y_train)
    lr_pred_proba_test = lr.predict_proba(X_test)[:, 1]
    lr_pred_proba_train = lr.predict_proba(X_train)[:, 1]

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
    cb_pred_proba_test = model.predict_proba(test_pool)[:, 1]
    cb_pred_proba_train = model.predict_proba(train_pool)[:, 1]

    final_pred_proba_train = 0.6 * cb_pred_proba_train + 0.4 * lr_pred_proba_train  # # тюнить веса
    final_pred_train = (final_pred_proba_train > 0.5).astype(int)             # # тюнить порог

    final_pred_proba_test = 0.6 * cb_pred_proba_test + 0.4 * lr_pred_proba_test  # # тюнить веса
    final_pred_test = (final_pred_proba_test > 0.5).astype(int)             # # тюнить порог
    return final_pred_test, final_pred_train, lr, model, y_test, y_train


@app.cell
def _(
    accuracy_score,
    f1_score,
    final_pred_test,
    final_pred_train,
    pd,
    y_test,
    y_train,
):
    df=pd.DataFrame(
        {"Какая выборка?": ["Выборка train","Выборка test"],
         "Accuracy метрика": [accuracy_score(y_train, final_pred_train),accuracy_score(y_test,final_pred_test)],
         "F1-мера": [f1_score(y_train, final_pred_train),f1_score(y_test,final_pred_test)]
            }
    )
    print(df)
    return


@app.cell
def _(auc, final_pred_test, plt, roc_curve, y_test):
    #ROC-AUC
    fpr_cb, tpr_cb, _ = roc_curve(y_test,final_pred_test) #false positive rate (доля ложноположительных срабатываний), true positive rate (чувствительность recall)
    roc_auc_cb = auc(fpr_cb, tpr_cb) #AUC = 1 — идеальная классификация, 0.5 — случайная

    plt.plot(fpr_cb, tpr_cb, label=f'Ансамбль Logreg Catboost (AUC = {roc_auc_cb:.3f})')

    plt.plot([0,1],[0,1],'--') ## Эта линия соответствует случайной модели (AUC=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('image/logreg+catboost_roc_curve.png')
    plt.show()
    return


@app.cell
def _(
    confusion_matrix,
    final_pred_test,
    final_pred_train,
    pd,
    precision_score,
    recall_score,
    y_test,
    y_train,
):
    cm = confusion_matrix(y_test,final_pred_test) #матрица неточностей, показывающая TP, FN, TN, FP
    cm_df = pd.DataFrame(
        cm,
        index=['Negative', 'Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    print(cm_df)
    print(f"Precission for train: {precision_score(y_train, final_pred_train):.3f}\nRecall for train: {recall_score(y_train, final_pred_train):.3f}\nPrecission for test: {precision_score(y_test,final_pred_test):.3f}\nRecall for test: {recall_score(y_test,final_pred_test):.3f}")
    return


@app.cell
def _(lr, model, re):
    #catboost напрямую не дает посмотреть важность слов, но дает оценить важность стобцов
    fi = model.get_feature_importance(prettified=True)
    print(fi.head(10))

    feature_names = lr.named_steps['preprocessor'].get_feature_names_out()
    coef = lr.named_steps['classifier'].coef_[0]

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
    #Итого
    Ансамбль не дал заметного прироста

    ##Результаты
    |Метрика|Train|Test|
    |:-----:|:----:|:--:|
    |Accuracy|0.836873|0.821882|
    |F1-мера|0.837470|0.822726|
    |Recall|0.841|0.827|
    |Precission|0.834|0.819|

    ##Матрица неточностей
    ||Прогноз y>0|Прогноз y<0|
    |:-----:|:----:|:--:|
    |Реал y>0|131969|27651|
    |Реал y<0|29220|130449|

    , где TP = 131969, TN = 130449, FP = 29220, FN = 27651

    Слова имеющие наибольший вес в модели:

    |Позитивные слова: вес|Негативные слова: вес|
    |:-----:|:----:|
    |thanks: 4.4|sad: -11.1|
    |isnt bad: 4.5|sadly: -6.6|
    |thank: 4.5|poor: -6.5|
    |dont feel bad: 4.7|bummed: -6.5|
    |wish luck: 5.6|miss: -6.2|
    """)
    return


@app.cell
def _(lr, model, pickle):
    package = {
        # Модели
        'model': model,
        'logistic_pipeline': lr,  # Pipeline с TF-IDF и OneHotEncoder
    
        # Параметры для ансамбля
        'ensemble_weights': {'catboost': 0.6, 'logistic': 0.4},
        'threshold': 0.5,
    
        # Признаки
        'cat_features': ['day', 'time'],
        'text_features': ['text'],
        'text_truncate': 150,
    
        # Дополнительные признаки, которые добавлялись
        'engineered_features': ['text_len', 'word_count', 'unique_word_ratio']
    }

    # Сохраняем одним файлом
    with open('models/ensemble.pkl', 'wb') as f:
        pickle.dump(package, f)
    return


if __name__ == "__main__":
    app.run()
