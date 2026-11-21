Детали на конвейере · parts-v2
Название команды: Котята Дейкстрята
Имена, группы, аккаунты Kaggle, вклад участников: 
+ Юрий Магус, 23КНТ-7, [Kaggle](https://www.kaggle.com/yuriymagus). Проанализировал данные и ошибки исходной CNN, провёл эксперименты с архитектурой и преобразованиями изображений. Сравнил модели по Macro-F1 и устойчивости на валидационной выборке, подготовил предсказания выбранной модели.
+ Лев Пестов, 23КНТ-6, [Kaggle](https://www.kaggle.com/levpestov). Сверил отчёт с критериями оценивания, привёл ноутбук и комплект файлов для сдачи в соответствие с требованиями преподавателя.
Kaggle ID посылки: 58479212

Рецепт: r3_normalized | seed: 17 | данные: parts-v2
Архитектура: shape_normalized | все параметры: 607976
Эпох: 36 | checkpoint: 30 | validation Macro-F1: 1.000000
Изображение: 64×64 | grayscale: True
На train: geometry=True, invert=True, weighted=False, blur=False, contrast=False.
Checkpoint loss tiebreak: True.
В solution.ipynb параметры преобразований заданы в коде обучения. Раздел «Итоговые результаты и решение» содержит таблицы запусков, кривые обучения, матрицы ошибок и метрики по классам. Конкретные ошибки и стоимость обучения разобраны в разделе «Анализ ошибок и вычислительная стоимость».
Остальная конфигурация обучения находится в metrics.json.
Предобучение: random
Обучение: Tesla T4 | AMP: True | время с валидацией: 62.94 сек.
Предсказание всех 960 изображений: CPU FP32, 53.70 сек. с чтением и записью файлов.

Python 3.13.16 | CUDA torch build: 13.0
Фактические версии:
torch==2.11.0+cu130
numpy==2.1.3
pandas==2.3.3
Pillow==13.0.0
matplotlib==3.10.0
ipython==7.34.0
torchvision==0.26.0+cu130
ipykernel==6.17.1
nbconvert==6.4.5

Обучение с нуля: загрузить solution.ipynb на Kaggle, подключить исходный parts-v2,
выбрать GPU T4 x2 и выполнить Save & Run All с RUN_MODE=full, RUN_ROUND=3.
Локальное обучение из каталога комплекта (PowerShell). Укажите свой путь к исходным CSV и images/:
$env:PARTS_DATA_ROOT = "D:\path\to\parts-v2"
$env:PARTS_RUN_MODE = "full"
$env:PARTS_ROUND = "3"
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output solution.executed.ipynb solution.ipynb
При установке зависимостей локально выберите совместимые сборки torch и torchvision для своего устройства.
Финальная модель обучается с нуля. Для отдельного опыта с ResNet18 в первой серии можно подключить официальный resnet18-f37072fd.pth из torchvision и задать PARTS_RESNET_WEIGHTS_PATH.
Для отдельного воспроизведения CSV из этого каталога выполните:
python predict.py --data /path/to/parts-v2 --weights model.pt --settings model_config.json --output reproduced.csv

Файл submission.csv | SHA-256 012d8a09bed8858a470382adc0db3b30d618300be83939a467b1a563f3e91c05
Проверка: reproduced.csv побайтово совпал с экспортом.
Источники: baseline.py и predict_baseline.py Семенова Д. П., практики 03 и 04–05.
Использованы исходная SmallCNN и протокол валидации. Остаточные блоки и поздний pooling
адаптированы к 64×64. Таблицы экспериментов, аугментации и экспорт реализованы здесь.
