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
    from sklearn.metrics import accuracy_score, f1_score

    from catboost import CatBoostClassifier, Pool

    return (
        CatBoostClassifier,
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        Pool,
        TfidfVectorizer,
        accuracy_score,
        f1_score,
        pd,
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
    return final_pred_test, final_pred_train, y_test, y_train


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
    pd.DataFrame(
        {"Какая выборка?": ["Выборка train","Выборка test"],
         "Accuracy метрика": [accuracy_score(y_train, final_pred_train),accuracy_score(y_test,final_pred_test)],
         "F1-мера": [f1_score(y_train, final_pred_train),f1_score(y_test,final_pred_test)]
            }
    )
    return


if __name__ == "__main__":
    app.run()
