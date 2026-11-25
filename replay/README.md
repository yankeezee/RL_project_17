# Grouped replay: часть Anastasiia Ianke

Самостоятельный модуль по проекту **Does prioritized replay oversample observation outages in ViZDoom?**
В доступной папке отсутствовал код команды. Реализованы replay, проверки, диагностика,
frozen-buffer analysis и адаптер обновления learner. ViZDoom-обучение здесь не выполнялось.

## Быстрый запуск

Откройте терминал в этой папке. Локально проверено на Python 3.13.11, NumPy 2.4.2,
pytest 9.0.3, Matplotlib 3.10.8, PyTorch 2.10.0+cpu.
Для replay нужен только NumPy. Остальные зависимости нужны соответствующим задачам.

```powershell
python -m pip install -e ".[analysis,test,learner]"
python -m pytest -q
python -m scripts.synthetic_demo --output results/my_synthetic_run
python -m scripts.plot_diagnostics --logs results/my_synthetic_run/logs --frozen-analysis results/my_synthetic_run/frozen_analysis --output results/my_synthetic_run/figures
python -m scripts.benchmark --output results/my_benchmark.json
```

Демонстрация создаёт JSONL с флагом `synthetic=true`, frozen CSV/metadata,
точные вероятности четырёх методов и рисунки с явной подписью SYNTHETIC.
Для повторного запуска demo используйте новую выходную папку: logger защищает существующие логи.
Проверки запускались из исходной папки без установки пакета; команду установки здесь не запускали.

## Что передать команде

| Файл | Назначение |
|---|---|
| `outage_replay/buffer.py` | FIFO, четыре sampler, вероятности, веса, TD priorities, injected flags |
| `outage_replay/diagnostics.py` | batch/per-draw JSONL, TD и replay counts, weighted Huber по группам |
| `outage_replay/frozen.py` | экспорт общего буфера, перерасчёт четырёх методов, SHA-256 |
| `examples/learner_integration.py` | минимальный адаптер Double DQN optimizer step |
| `tests/` | формулы, частоты, перезапись, weights, logging и optimizer smoke tests |
| `scripts/` | synthetic demo, анализ CSV, графики реальных логов, benchmark |
| `configs/protocol.json` | параметры proposal и явно выбранные технические соглашения |
| `docs/INTEGRATION.md` | контракт для Areg и Leon, порядок действий |
| `docs/LOG_SCHEMA.md` | определения метрик и знаменателей |
| `docs/METHODS.md` | Methods на английском для статьи |
| `docs/AUDIT.md` | аудит, допущения и расхождения |
| `docs/WORKLOG.md` | выполненные команды, реальные результаты и ограничения |
| `RESEARCH_LOG.md` | пять циклов улучшений: проблемы, проверки и сравнения |
| `NEXT_STEPS.md` | актуальное состояние и команды для продолжения командой |
| `results/` | сохранённые результаты локальных проверок и синтетические артефакты |

## Существенные решения

Выборка независимая, с возвращением. Группа выбирается по текущей доле переходов,
затем переход по приоритету внутри неё. Это сохраняет долю **в ожидании**, а не в каждом batch.
Начальный raw priority пустого буфера равен 1.0; последующие вставки получают текущий максимум.
Повторяющиеся индексы обновляются по максимуму абсолютных TD errors.
Нормировка weights использует минимум вероятности по всему буферу, включая переходы вне batch.
Идентификатор перехода защищает от ошибочного обновления слота после FIFO-перезаписи.

Сумма `mean(w * Huber)` применяется одинаково для всех методов; делить на `sum(w)` нельзя.
Полный бюджет, архитектуру и наблюдения определяет общий learner команды.
Подключение к реальному коду и paired evaluation остаются зависимостями команды.

## Уже полученные результаты

Итоговый запуск после пяти циклов: **100 тестов прошли за 13.75 s**,
без failures/errors/skips; отчёты `results/improvements/final_pytest.txt` и `.xml`.
Первая версия: 65 тестов, отчёты сохранены в `results/pytest.txt` и `results/pytest.xml`.
Частотные тесты делают по 200 000 выборок на метод с фиксированным seed и
Bernstein-допуском с family-wise уровнем ошибки 1e-6 на пять категорий для каждого метода.

В синтетическом frozen buffer N=1000, 800 clean переходов имеют TD=1,
200 outage переходов имеют TD=100. Ожидаемая outage-масса:
uniform 0.20, PER 0.798477726, mixture 0.499238863, grouped 0.20.
Этот пример проверяет механизм и не подтверждает улучшение hit rate.

Benchmark первой версии: grouped sampling median 0.430 ms,
priority update median 5.303 ms, 6 892 896 bytes ndarray metadata.
Это стоимость этой реализации на одной машине, без среды и learner;
разница между методами не является надёжным performance-сравнением.

После улучшения batch refresh локальный paired benchmark на идентичных inputs:
legacy median 4.014 ms -> batched 0.614 ms (6.54x), деревья побитово идентичны.
Frozen analysis на 50000 transitions: 3.619 -> 0.656 s (5.51x), CSV/JSON
побайтово идентичны. Результаты: `results/improvements/`.
Добавлены replay checkpoints с RNG и finite checks до optimizer.step.
Новые поля логов отделяют exact expected sampler share от observed batch share;
старые логи по-прежнему поддерживаются графиками.

Основная гипотеза остаётся открытой до paired evaluation: grouped минус PER по hit rate
после burst-training p=.25, L=4 на held-out bursts L=8. Нужны также реальные свидетельства
избыточного outage sampling у PER; частота replay не равна влиянию на обучение.
