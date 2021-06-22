1. Для обработки датасета PPG_Dalia и получения результатов по  HR и HAR: 
    1. Скачать датасет: `https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA` 
    2. Запустить скрипт dalia_process.py:  
        `python polyn_process.py /path/to/raw/data /path/to/out/folder` 
    3. Запустить скрипт train_HAR.py: 
        `python train_HAR.py --path /data/ppg_release --epochs 3 --batch_size 128 --window_size 320 --step_size 32 --dropout_rate 0.5 --channels 3 64 64 128 256 512 8 --name "HAR ResNet" --nolog`
    4. Запустить скрипт train_HR.py: 
        `python train_HR.py --path /data/ppg_release --epochs 10 --batch_size 128 --window_size 640 --step_size 64 —dropout_rate 0.5 --channels 4 64 64 128 256 512 1 --name "HR ResNet" --nolog`

2. Для обработки датасета  crossfit и получения результатов для дескрипторов:
    1. Скачать данные: `https://drive.google.com/drive/folders/1BvjO8YxJjk5BPZAcEO42HuyJhja6XOOf`
     (HAR_Crossfit_Sensors_Data/data/constrained_workout/preprocessed_numpy_data/np_exercise_data - путь до нужных данных)
    2. Запустить скрипт crossfil_process.py: `python crossfil_process.py`
    3. Запустить скрипт train_DE.py
