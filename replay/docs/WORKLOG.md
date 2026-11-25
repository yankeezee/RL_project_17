# Журнал выполнения

Дата: 25 ноября 2026 (Asia/Irkutsk). Проект создан отдельно от учебных материалов.

Дополнение после первой версии: выполнены пять циклов, подробно в
`../RESEARCH_LOG.md`, актуальные зависимости в `../NEXT_STEPS.md`.
Последний полный запуск: **100 passed in 13.75s**; исходные 65-test результаты
ниже сохранены как история. Добавлен replay state checkpoint, но полный learner
resume по-прежнему зависит от команды. Новые benchmarks: priority refresh 6.54x,
frozen analysis 5.51x локально, при идентичных результатах. Артефакты:
`results/improvements/final_pytest.*`, `cycle4_benchmark.json`, `frozen_benchmark/comparison.json`.

## Выполнено и проверено

1. Прочитан запрос и обе страницы приложенного proposal, просмотрены их PNG.
   В workspace нет исходного learner/replay/training/evaluation кода.
2. Созданы grouped replay, uniform/PER/mixture probability references,
   global importance weights, FIFO, current-max priority initialization,
   duplicate reduction, transition IDs, injected flag tagging, beta schedule.
3. `python -m pytest -q`: **55 passed in 2.96s** до learner adapter.
   Включены 4 frequency tests по 200000 draws каждый; seed 1037.
   Для пяти категорий каждого метода используется union Bernstein bound,
   alpha=1e-6; допуск зависит от p, Ndraws и числа категорий, не от желаемого результата.
4. Добавлены logger, frozen analysis и адаптер к существующим сетям/optimizer.
   После добавления integration tests: **61 passed in 11.98s**.
   Реально выполнены Adam steps и backprop для всех четырёх sampler на CPU.
   Отдельно проверены reward-only targets при terminal и timeout, даже если
   target network даёт NaN для next states: она не вызывается для task end.
5. `python -m scripts.synthetic_demo`: завершился успешно.
   Общий frozen buffer: N=1000, outage=.2, clean TD=1, outage TD=100.
   Ожидаемые outage masses: uniform=.2, PER=.7984777260370877,
   mixture=.4992388630185438, grouped=.2.
   Это искусственно заданные residuals, **не результат ViZDoom**.
6. Dynamic synthetic stream: по 200 batch, 12800 draws на метод, capacity=256,
   FIFO с одинаковым входным stream и независимым replay RNG.
   Для PER outage draws=7046 против time-matched occupancy baseline=2930;
   grouped=2902 против 2930. Это проверка диагностического pipeline,
   не оценка улучшения обучаемого агента; в demo optimizer не запускался.
7. `python -m scripts.plot_diagnostics --logs results/synthetic/logs --frozen-analysis results/synthetic/frozen_analysis --output results/synthetic/figures`:
   создано 5 рисунков и 4 allocation JSON. Frozen/grouped графики визуально проверены:
   читаемые оси и подписи synthetic, раздельные loss denominators, time-matched shares.
8. `python -m scripts.benchmark`: capacity=50000, batch=64, 20 warm-up
   и 300 измеренных batch на метод, seed=20261125. Результат: `results/benchmark.json`.
9. Добавлены edge-case tests: глобальный минимум grouped probability может находиться
   в другой группе, чем минимум priority mass; нулевые TD errors; сброс count при FIFO;
   batch-only logging и blank TD в frozen CSV. Финальный запуск
   `python -m pytest -q --junitxml=results/pytest.xml`: **65 passed in 11.27s**,
   без failures/errors/skips. Сохранены `results/pytest.txt`, `pytest.xml`, `validation.json`.

| Метод | Sampling median, ms | Sampling p95, ms | Priority-update median, ms |
|---|---:|---:|---:|
| uniform | 0.172 | 0.261 | 5.590 |
| PER | 0.451 | 0.881 | 5.384 |
| mixture | 0.413 | 0.692 | 5.202 |
| grouped | 0.430 | 0.786 | 5.303 |

Ndarray metadata: **6 892 896 bytes**, около 6.57 MiB, без payload/Python/RSS.
Batch+per-draw logging median=1.521 ms, p95=3.047 ms,
около 27360 bytes/update. Фоновая активность и последовательный порядок методов
ограничивают сравнение скорости; benchmark является локальным smoke measurement.
Для main training разумно одинаково ограничить per-draw diagnostic windows,
так как подробный JSONL на всём бюджете может занимать гигабайты на run.

## Ошибки инструментов и исправления

- В bundled Poppler есть pdftoppm, но нет pdftotext по ожидаемому пути.
  PDF извлечён bundled pypdf; PNG созданы pdftoppm.
- Печать извлечённого Unicode-текста из bundled Python упёрлась в cp1251 консоли.
  UTF-8 файл был успешно записан; прочитан через PowerShell с явным UTF-8.
- Тесты реализации не обнаружили failures в первых двух запусках.
- В добавленном тесте frozen export была ошибочная проверка `replay_count=0`
  после двух draws. Запуск дал 1 failed / 64 passed. Проверка исправлена на
  чтение отдельных CSV-полей: blank td_abs и count=2. Повторный полный запуск
  дал 65 passed. Код sampler/export при этом менять не потребовалось.

## Принятые допущения

С возвращением; max abs TD для дубликатов; начальный raw=1; current max до eviction;
pre-step TD refresh после step; unit-threshold Huber; loss mean over batch;
однопоточный доступ и caller-owned immutable payloads. Подробно: `AUDIT.md`.
Эти решения не изменяют архитектуру, reward, observation или update budget.

## Оставшиеся зависимости и непроверенные утверждения

- Реальный репозиторий команды и сравнение с готовым PER отсутствуют.
- Нужны environment/delivery/timeout hooks Leon и learner hooks Areg.
- Реальные ViZDoom masks, feature histories, remaining-time feature, обучение
  4x2x3 runs и final checkpoints здесь не проверены и не выполнены.
- GPU, распределённая/async работа и восстановление learner checkpoints не проверены.
- Frozen snapshot не хранит transitions/сети/RNG и не является resumable checkpoint.
  Для свежих общих TD errors нужны выбранная frozen сеть и доступ к payloads.
- Не проводились hit-rate evaluation, hierarchical bootstrap и проверка гипотезы
  на реальном опыте. Корректный sampler не доказывает пользу grouping.
- Для поддержки гипотезы нужны положительный primary contrast и excess PER
  allocation на реальных логах; CI через ноль оставляет performance claim открытым.
  Сопоставимый эффект mixture ослабляет аргумент в пользу специфического grouping.
- Три training seeds ограничивают точность неопределённости; обобщение за пределы
  predict_position потребует других экспериментов.
