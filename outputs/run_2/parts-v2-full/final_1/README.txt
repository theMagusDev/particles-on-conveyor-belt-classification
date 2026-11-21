Детали на конвейере · parts-v2
Название команды: [заполнить]
Имена, группы, аккаунты Kaggle, вклад участников: [заполнить]
Kaggle ID посылки: [заполнить после загрузки]

Рецепт: gray_long | seed: 17 | данные: parts-v2
Архитектура: shape_cnn | все параметры: 607976
Эпох: 36 | checkpoint: 24 | validation Macro-F1: 1.000000
Изображение: 64×64 | grayscale: True
На train: geometry=True, invert=False, weighted=False.
Точные параметры преобразований и таблица опытов находятся в solution.ipynb.
Остальная конфигурация обучения находится в metrics.json.
Предобучение: random
Обучение: Tesla T4 | AMP: True | время с валидацией: 55.25 сек.
Предсказание всех 960 изображений: CPU FP32, 70.41 сек. с чтением и записью файлов.

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
выбрать GPU T4 x2 и выполнить Save & Run All с RUN_MODE=full.
Локальный запуск: python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 solution.ipynb
с PARTS_RUN_MODE=full и PARTS_DATA_ROOT, указывающей на каталог с CSV и images/.
При установке зависимостей локально выберите совместимые сборки torch и torchvision для своего устройства.
Для офлайн-обучения ResNet18 подключите официальный resnet18-f37072fd.pth из torchvision
или задайте PARTS_RESNET_WEIGHTS_PATH с путём к этому файлу.
Для отдельного воспроизведения CSV из этого каталога выполните:
python predict.py --data /path/to/parts-v2 --weights model.pt --settings model_config.json --output reproduced.csv

Файл submission.csv | SHA-256 bd330795da56d368f75877269eaa63928cb8bc5f97a389af13921fc0d1ea6b2c
Проверка: reproduced.csv побайтово совпал с экспортом.
Источники: baseline.py и predict_baseline.py Семенова Д. П., практики 03 и 04–05.
Использованы исходная SmallCNN и протокол валидации. Остаточные блоки и поздний pooling
адаптированы к 64×64. Таблицы экспериментов, аугментации и экспорт реализованы здесь.
