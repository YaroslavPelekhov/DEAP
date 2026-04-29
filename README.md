# NeuroBarometer — Emotion Recognition

Распознавание эмоций (валентность / активация) по физиологическим сигналам:
**EEG + PPG + GSR**.  
Датасет: [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — 32 испытуемых, 40 видео-трейлов.

## Быстрый старт

```bash
pip install -r requirements.txt

# Обучение (SD, 32 субъекта)
python scripts/train.py --protocol sd --model multimodal

# Ablation по модальностям
python experiments/ablation_modalities.py

# Инференс на данных с барометра
python experiments/barometer_inference.py \
    --model data/models/loso_s01_multimodal.pt \
    --eeg barometer_eeg.npy --ppg barometer_ppg.npy --gsr barometer_gsr.npy
```

## Структура проекта

```
src/
  data/channels.py       — DEAP → 10-20 имена, каналы барометра (19 общих)
  features/
    eeg.py               — Differential Entropy + Hjorth (32ch × 6 = 192)
    ppg.py               — HRV: 10 признаков
    gsr.py               — EDA: 8 признаков (без DEAP-специфичного консенсуса)
    pipeline.py          — FeaturePipeline с кэшированием
  models/
    multimodal.py        — MultiModalNet (раздельные энкодеры + Modality Dropout)
    temporal.py          — TemporalNet (Bi-GRU, моделирует динамику трейла)
    mmcat.py             — MMCAT Transformer (cross-attention EEG↔PPG+GSR)
    factory.py           — create_model(name, in_eeg, in_ppg, in_gsr)
  training/
    trainer.py           — train_sd() / train_loso()
    metrics.py           — accuracy, F1, majority vote
  utils/
    io.py                — save/load модели (.pt) + результаты (.json)
data/
  raw/          → .gitignore (DEAP .dat, скачать с сайта)
  features/     → .gitignore (кэш признаков, ~67 MB)
  models/       — веса обученных моделей
experiments/
  ablation_modalities.py — таблица точности по комбинациям модальностей
  barometer_inference.py — инференс на данных с устройства
```

## Каналы барометра

Барометр имеет 20 ЭЭГ каналов (старая нотация 10-20):
```
Fp1 F7 F8 T4 T6 T5 T3 Fp2 O1 P3 Pz F3 Fz F4 C4 P4 POz C3 Cz O2
```
Из них **19 совпадают с DEAP** (POz отсутствует в DEAP). 
Пересечение (в новой нотации): `Fp1 F7 F8 T8 P8 P7 T7 Fp2 O1 P3 Pz F3 Fz F4 C4 P4 C3 Cz O2`

## Данные

Сырой датасет DEAP требует регистрации:  
→ http://www.eecs.qmul.ac.uk/mmv/datasets/deap/  
Разместите файлы `s01.dat … s32.dat` в папку, указанную в `config.py` как `DATA_DIR`.
