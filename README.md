# Детали на конвейере

Решение учебного [соревнования на Kaggle](https://www.kaggle.com/competitions/parts-on-conveyor-semenov/overview). По фото 64×64 определяем, какая из восьми деталей лежит на конвейере.

Начали с простой CNN, потом попробовали позже уменьшать изображение, менять аугментации и нормализацию. В итоге выбрали свою CNN с 608 тыс. параметров. Она правильно распознала все 480 деталей на валидации (Macro-F1 1.000). На публичной части Kaggle у посылки № 58479212 тоже 1.000.

В [ноутбуке](solution.ipynb) есть код обучения, сравнение экспериментов и разбор ошибок. Веса и файлы для сдачи лежат в [финальной папке](outputs/submission_final_1/).

## Как получить предсказания

Из корня проекта, если установлен `uv`:

```bash
uv run --no-project --python 3.13 --with torch==2.11.0 --with numpy==2.1.3 \
  --with pandas==2.3.3 --with Pillow==13.0.0 python predict.py --data data \
  --weights outputs/submission_final_1/model.pt \
  --settings outputs/submission_final_1/model_config.json --output reproduced.csv
```

Скрипт работает на CPU и создаёт `reproduced.csv` из сохранённых весов.
