# NeuroBarometer — Emotion Recognition

Распознавание эмоций (валентность / активация) по физиологическим сигналам:
**EEG + PPG + GSR**.

Датасет для обучения: [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — 32 испытуемых, 40 видео-трейлов.
Целевое устройство: NeuroBarometer (20-канальный EEG + GSR + PPG).

---

## Структура проекта

```
├── config.py                    # все пути и гиперпараметры
├── requirements.txt
│
├── src/                         # основные модули
│   ├── data/
│   │   ├── loader.py            # загрузка .dat файлов DEAP
│   │   └── channels.py          # имена каналов в системе 10-20,
│   │                            # пересечение DEAP ↔ Барометр (19 каналов)
│   ├── features/
│   │   ├── eeg.py               # EEGExtractor: DE по полосам + Hjorth (32ch × 6 = 192)
│   │   ├── ppg.py               # 10 HRV-признаков
│   │   ├── gsr.py               # 8 EDA-признаков (без DEAP-специфичного консенсуса)
│   │   └── pipeline.py          # FeaturePipeline: запуск + кэширование
│   ├── models/
│   │   ├── multimodal.py        # MultiModalNet — раздельные энкодеры + Modality Dropout
│   │   ├── temporal.py          # TemporalNet — Bi-GRU по последовательности окон
│   │   ├── mmcat.py             # MMCAT — MultiModal Cross-Attention Transformer
│   │   └── factory.py           # create_model(name, in_eeg, in_ppg, in_gsr)
│   ├── training/
│   │   ├── trainer.py           # train_sd() / train_loso(), сохранение весов
│   │   └── metrics.py           # accuracy, F1, majority vote
│   └── utils/
│       └── io.py                # save/load модели (.pt) + результаты (.json)
│
├── data/
│   ├── raw/                     # s01.dat … s32.dat (DEAP, 32 субъекта)
│   ├── features/                # кэш признаков (генерируется автоматически)
│   └── models/                  # веса обученных моделей (.pt)
│
├── experiments/
│   ├── ablation_modalities.py   # таблица точности по 7 комбинациям {EEG/PPG/GSR}
│   └── barometer_inference.py   # инференс на данных с устройства NeuroBarometer
│
└── scripts/
    └── train.py                 # CLI: обучение SD / LOSO
```

---

## Быстрый старт

```bash
pip install -r requirements.txt

# Обучение (subject-dependent, все 32 субъекта)
python scripts/train.py --protocol sd --model multimodal --save

# LOSO
python scripts/train.py --protocol loso --model temporal --save

# Ablation: точность при разных комбинациях модальностей
python experiments/ablation_modalities.py

# Инференс на данных с барометра
python experiments/barometer_inference.py \
    --model data/models/loso_s01_multimodal.pt \
    --eeg eeg.npy --ppg ppg.npy --gsr gsr.npy
```

---

## Модели

| Модель | Описание | Вход |
|--------|----------|------|
| `multimodal` | Раздельные MLP-энкодеры + Modality Dropout | окна (B, feats) |
| `temporal` | Bi-GRU по последовательности 60 окон трейла | трейлы (B, T, feats) |
| `mmcat` | Cross-Attention Transformer EEG ↔ PPG+GSR | окна (B, feats) |

Все модели принимают переменный размер входа — `create_model(name, in_eeg=192, in_ppg=10, in_gsr=10)`.

**Modality Dropout**: при обучении каждая модальность случайно обнуляется с p=0.2.
На инференсе можно подавать любое подмножество (например, только EEG+GSR без PPG).

---

## Признаки

### EEG — 192 признака (32 канала × 6)
| Признак | Описание |
|---------|----------|
| DE theta (5–7 Гц) | Дифференциальная энтропия |
| DE alpha (8–13 Гц) | Дифференциальная энтропия |
| DE beta (14–30 Гц) | Дифференциальная энтропия |
| DE gamma (31–45 Гц) | Дифференциальная энтропия |
| Hjorth Mobility | Временной дескриптор |
| Hjorth Complexity | Временной дескриптор |

Скользящие окна 1 с, baseline-коррекция (первые 3 с трейла).

### PPG — 10 признаков
SDNN, RMSSD, pNN50, HR, LF, HF, LF/HF, mean_amp, std_amp, IBI_CV

### GSR — 10 признаков
SCL_mean, SCL_std, EDA_mean, EDA_std, N_peaks, peak_amp, rise_rate, slope, FAA, FTA

> FAA (Frontal Alpha Asymmetry) и FTA (Frontal Theta Asymmetry) — физиологические признаки,
> вычисляются из каналов F3/F4. DEAP-специфичный консенсус убран (не переносится на новые стимулы).

---

## Каналы NeuroBarometer

Барометр: 20 каналов (старая нотация 10-20):
```
Fp1  F7  F8  T4  T6  T5  T3  Fp2
O1   P3  Pz  F3  Fz  F4  C4  P4
POz  C3  Cz  O2
```

Соответствие новой нотации: T3→T7, T4→T8, T5→P7, T6→P8.
**19 каналов совпадают с DEAP** (POz отсутствует в DEAP-32):
```
Fp1  F7  F8  T8  P8  P7  T7  Fp2  O1
P3   Pz  F3  Fz  F4  C4  P4  C3   Cz  O2
```

При инференсе на барометре подавайте каналы именно в этом порядке.

---

## Протоколы обучения

**Subject-Dependent (SD)** — GroupKFold по трейлам (8 фолдов, 5 трейлов на фолд).
Majority voting: агрегация предсказаний по окнам → предсказание на трейл.

**LOSO** — Leave-One-Subject-Out. Тренировка на 31, тест на 1.

---

## Результаты (MultiModalNet, v13)

| Протокол | Valence Acc | Arousal Acc |
|----------|-------------|-------------|
| SD       | ~82 %       | ~71 %       |

*Результаты по всем субъектам сохраняются в `results/` после запуска.*
