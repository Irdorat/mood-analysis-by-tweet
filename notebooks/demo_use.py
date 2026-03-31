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
    #Демо использование обученной модели на данных пользователя
    Показать работу модели на примере пользовательского твита. Модель принимает твит пользователя, а также текущую дату.
    """)
    return


@app.cell
def _():
    import pickle
    from datetime import datetime
    import numpy as np
    import pandas as pd

    return datetime, pd, pickle


@app.cell
def _(datetime, pd, pickle):
    with open('models/catboost.pkl', 'rb') as f:
        model = pickle.load(f)

    model = model['catboost_model']  # или другой ключ, под которым сохранена модель
    
    now = datetime.now()

    day=now.strftime('%a')
    hour=now.hour
    def get_time(hour):
        if 6<=hour<11:
            return "morning"
        elif 11<=hour<16:
            return "day"
        elif 16<=21:
            return "evening"
        else:
            return "night"
    time=get_time(hour)
    text=input('Пользовательский твит: ')
    text_len = len(text)
    word_count = len(text.split())
    words = text.split()
    if words:
        unique_word_ratio = len(set(words)) / (len(words) + 1)
    else:
        unique_word_ratio = 0
    
    user_df=pd.DataFrame({
        'text': [text],
        'day': [day],
        'time': [time],
        'text_len': [text_len],
        'word_count': [word_count], 
        'unique_word_ratio': [unique_word_ratio]
    })
    y_user = model.predict(user_df)[0]
    print(f'Прогноз эмоциональности твита: {"Негативный" if y_user==0 else "Положительный"}')
    return


if __name__ == "__main__":
    app.run()
