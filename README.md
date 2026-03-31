## Emotion Classification in Tweets

Python | Pandas | Scikit-learn | CatBoost 

Проект посвящён классификации эмоциональной окраски твитов.
Цель - построить модель, которая на основе текста и вспомогательных признаков (день недели, длина текста, время) предсказывает позитив/негатив (1,0) твита.

<h1> Tech Stack <a href="#-tech-stack--"><img src="https://raw.githubusercontent.com/HighAmbition211/HighAmbition211/auxiliary/others/skill.gif" width="32"></a> </h1>

### Languages
<table>
  <tr>
    <td align="center" width="90">
      <a href="https://www.python.org/" target="_blank">
        <img alt="Python" width="45" height="45" src="https://raw.githubusercontent.com/HighAmbition211/HighAmbition211/auxiliary/languages/python.svg" />
      </a>
      <br><h4>Python</h4>
    </td>
  </tr>
</table>

### Frameworks & Libraries
<table>
  <tr>
    <td align="center" width="90">
      <a href="https://pandas.pydata.org/" target="_blank">
        <img alt="Pandas" width="45" height="45" src="https://pandas.pydata.org/static/img/pandas_mark.svg" />
      </a>
      <br><h4>Pandas</h4>
    </td>
    <td align="center" width="90">
      <a href="https://scikit-learn.org/" target="_blank">
        <img alt="Scikit-learn" width="45" height="45" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" />
      </a>
      <br><h4>Scikit-learn</h4>
    </td>
    <td align="center" width="90">
      <a href="https://catboost.ai/" target="_blank">
        <img alt="CatBoost" width="45" height="45" src="http://storage.mds.yandex.net/get-devtools-opensource/250854/catboost-logo.png" />
      </a>
      <br><h4>CatBoost</h4>
    </td>
  </tr>
</table>

## Описание проекта
Проект включает следующие шаги:

1. **Сбор и предобработка данных**
- Тексты твитов очищаются, обрезаются до 150 символов, нормализуются
- Категориальные данные (день недели и время суток) кодируется через OHE
- Текстовые данные обрабатываются через TF-IDF и CatBoost (text features)

2. **Обучение моделей**
- Задача - бинарная классификация (положительная или отрицательная полярность сообщения)
- Baseline - логистическая регрессия (использует TF-IDF и OHE)
- CatBoost - бустинг модель, хорошо работающая с категориальными и текстовыми признаками
- Ансамбль (catboost и логистическая регрессия). **Показала плохо и требует доработки**

3. **Оценка моделей**
- Метрики: accuracy, F1, Precission, Recall
- Визуализация: ROC-кривая, Матрица неточностей, Feature Importance

## Структура проекта
```
|src
|-models
||-catboost.pkl #обученная catboost модель
||-ensemble.pkl #обученная ensamble модель
||-logreg.pkl #обученная logistic regression модель
|-notebooks
||-ansamble_logreg_catboost #скрипт обучения ансамбля
||-baseline #скрипт обучения baseline модели
||-catboost_model #скрипт обучения catboost
||-catboost_tuningv2_model #скрипт настройки гиперпараметров модели catboost
||-demo_use #скрипт демо использования пользователем
||-eda #очистка и предобработка набора данных
|-requirements.txt
```
## Начало работы
1) Клонировать репозиторий
git clone https://github.com/Irdorat/mood-analysis-by-tweet

cd mood-analysis-by-tweet

2) Создать виртуальное окружение (из файла req*.txt)

python -m venv venv

venv\Scripts\activate

3) Установить зависимости

pip install -r requirements.txt

4) Использовать обученные модели в папке models

## Результаты

|Метрика|Logistic Regression|Catboost|Ensemble Catboost|
|:-----:|:-----------------:|:------:|:---------------:|
|Accuracy|0.788302|0.823605|0.821882|
|F1-мера |0.792749|0.823060|0.822726|
|Precission|0.776|0.825|0.819|
|Recall|0.810|0.821|0.827|

### Слова имеющие наибольший вес в модели:

|Позитивные слова: вес|Негативные слова: вес|
|:-----:|:----:|
|thanks: 4.4|sad: -11.1|
|isnt bad: 4.5|sadly: -6.6|
|thank: 4.5|poor: -6.5|
|dont feel bad: 4.7|bummed: -6.5|
|wish luck: 5.6|miss: -6.2|
---
