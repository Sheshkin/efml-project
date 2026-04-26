# Бенчмарк ускорения инференса UNet

Проект для сравнения методов ускорения инференса на задаче сегментации Oxford-IIIT Pet.

Core-эксперименты:
- Baseline FP16
- `torch.compile`
- ONNX Runtime
- PTQ (int8)
- Unstructured Pruning
- 2:4 Semi-structured Sparsity

Опциональные эксперименты:
- QAT (int8)

## Установка

```bash
pip install -r requirements.txt
```

Скачать датасет:

```bash
bash scripts/download_data.sh
```

## Запуск

```bash
python run_experiment.py <команда>
```

| Команда       | Описание |
|---------------|----------|
| `train`       | Обучение/дообучение базовой модели |
| `baseline`    | Бенчмарк Baseline FP16 |
| `compile`     | Бенчмарк `torch.compile` |
| `backend`     | Бенчмарк ONNX Runtime |
| `ptq`         | Пост-тренировочная квантизация (int8) + бенчмарк |
| `qat`         | QAT (fake quant + convert int8) + бенчмарк |
| `prune`       | Unstructured pruning + восстановительное дообучение |
| `semi24`      | 2:4 semi-structured sparsity + восстановительное дообучение |

## One-click pipeline для 6 core-экспериментов

```bash
python run_future_experiments.py
```

Fail-fast режим:

```bash
python run_future_experiments.py --fail-fast
```


```bash
python run_future_experiments.py --include-optional
```

Pipeline пишет:
- JSON-результаты: `results/logs/*_result.json`
- Manifest запуска: `results/logs/future_pipeline_manifest.json`
- Сводный JSON: `results/logs/summary_results.json`
- Таблицы: `results/tables/performance.md`, `results/tables/quality.md`
- Sanity-checks: `results/tables/validation.md`
- Графики: `results/plots/*.png`
чмарков нужен checkpoint: `results/checkpoints/best_model.pt`.

## Отчет

`Отчет.pdf`.
