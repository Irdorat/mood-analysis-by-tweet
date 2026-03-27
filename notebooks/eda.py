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
    #Определение цели и постановка задачи
        Цель: определение эмоциональной окраски твита

        Задача: применение методов классического ML для определения позитивной/негативной окраски
        небольшого фрагмента текста на основе его содержания и времени написания.

        Задача будет решаться путем бинарном классификации, а в качестве метрики выбираем log-loss, accuracy и f1-меру
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Анализ данных

        На этом этапе оцениваем структуру данных, качество признаков, распределение метки и определение
        потенциальных проблем перед обучением моделей
    """)
    return


@app.cell
def _():
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re

    return deepcopy, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Работа с датасетом

    Рассмотрим датасет и определим спектр работ созданию наборов данных
    """)
    return


@app.cell
def _(pd):
    data=pd.read_csv('data/raw/training_data.csv', delimiter = ',', quotechar='"', names = ['y','id message','date','flag','user','text'],encoding = 'latin1')
    data.head()
    return (data,)


@app.cell
def _(data):
    data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Пропусков данных нет

    Датасет состоит из следующих столбцов:

    1) y (таргет) - числовой признак принимает значения полярности сообщения (0 - негативный, 4 - позитивный)

    2) id message - числовой признак уникальный ID сообщения

    3) date - категориальный признак дата и время сообщения

    4) flag - категориальный признак запроса

    5) user - категориальный признак (ник пользователя)

    6) text - категориальный признак (сообщение)

    Следующий шаг посмотреть распределение таргета и постановка задачи в рамках pipeline
    """)
    return


@app.cell
def _(data):
    data['y'].value_counts() #данные разделились ровно 50 на 50, что позволяет использовать accuracy и f1-меру
    return


@app.cell
def _(data):
    data['flag'].value_counts() #бесполезный столбец
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Удаляем ненужные колонки, дорабатываем значения и подготовливаем датасет к обучению
    """)
    return


@app.cell
def _(data, deepcopy, pd, re):
    clean_data=deepcopy(data)
    clean_data['y']=clean_data['y'].map({0:0, 4:1}) #меняем 4 на 1
    clean_data['date']=clean_data['date'].str.replace(r'[A-Z]{3}',' ', regex=True) #убрали метку региона
    clean_data['date']=pd.to_datetime(clean_data['date'], format= '%a %b %d %H:%M:%S %Y') #переводим дату и время в более удобный формат
    clean_data['day']=clean_data['date'].dt.strftime('%a') #в отдельный стоблец выносим день недели
    clean_data['hour']=clean_data['date'].dt.hour
    def get_time(hour):
        if 6<=hour<11:
            return "morning"
        elif 11<=hour<16:
            return "day"
        elif 16<=21:
            return "evening"
        else:
            return "night"
    clean_data['time']=clean_data['hour'].apply(get_time) #заменили время на промежутки времени (утро, день, вечер и ночь)
    clean_data['text']=clean_data['text'].apply(lambda x: re.sub(r'(@\w+|http\S+)', '', x).strip()) #очистим текст от артефактов пользователей
    def clean_text(text):
        # нижний регистр
        text = text.lower()
        # убрать все спецсимволы, оставляем только буквы и пробелы
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # убрать лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    clean_data['text']=clean_data['text'].apply(clean_text)
    clean_data.drop(['id message','flag','user','hour','date'], axis=1, inplace=True)

    clean_data.head()
    return (clean_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Итого

    Результатом этого этапа является:

    - сформулированная задача (применение методов классического ML для определения позитивной/негативной окраски)
    - определены метрики модели (accuracy, f1-мера)
    - данные очищены, не содержат пропусков
    - распределение классов равномерно

    На следующих этапах будет построена baseline модель классификации
    """)
    return


@app.cell
def _(clean_data):
    clean_data.to_csv('data/processed/clean_data.csv', index=False, encoding='utf-8')
    return


if __name__ == "__main__":
    app.run()
