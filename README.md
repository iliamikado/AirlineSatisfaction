# Airline Satisfaction — Kaggle


## Setup

Place the competition files in `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Train and save the model:

```powershell
python train.py
```

Generate a Kaggle submission:

```powershell
python predict.py
```

The resulting file is saved to `outputs/submission.csv`.