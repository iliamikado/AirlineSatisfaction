# Airline Satisfaction — Kaggle

Стартовое решение для соревнования Kaggle [Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10).

Цель — предсказать вероятность того, что пассажир удовлетворён перелётом. Метрика соревнования — ROC-AUC.

## Что внутри

```text
.
├── data/                   # CSV-файлы соревнования; не хранится в Git
├── models/                 # Обученные модели; не хранится в Git
├── outputs/                # Kaggle submissions; не хранится в Git
├── train.py                # Обучение и локальная проверка ROC-AUC
├── predict.py              # Генерация submission.csv из сохранённой модели
└── requirements.txt        # Python-зависимости
```

## Установка

Нужен Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Данные

Скачай файлы соревнования и положи их в папку `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Эти файлы исключены из Git, поскольку соревнование распространяет их отдельно.

## Обучение

```powershell
python train.py
```

Скрипт разделит `train.csv` на обучающую и проверочную части, выведет ROC-AUC и сохранит CatBoost-модель в `models/catboost_baseline.cbm`.

## Submission для Kaggle

```powershell
python predict.py
```

Скрипт создаст `outputs/submission.csv`. Его можно загрузить на странице соревнования через **Submit Predictions**.

Файл имеет вид:

```csv
id,satisfaction
699635,0.249919
699636,0.896314
```

`satisfaction` — вероятность от `0` до `1`, а не значение `True`/`False`.

## Воспроизводимость

В `train.py` задан `RANDOM_STATE = 42`, поэтому при тех же данных и окружении разбиение данных воспроизводимо.
