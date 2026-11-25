# Текущее состояние и продолжение

**Завершены все 5 циклов. Итог: 100 passed in 13.75s, без пропусков.**
Исследовательские формулы, budget, метрики и primary comparison неизменны.

Текущее состояние:

- Invalid TD/loss/gradients отклоняются до optimizer.step.
- Replay state_dict восстанавливает RNG, payloads, FIFO, priorities и counts.
- Логи сохраняют exact expected sampler allocation на момент sampling.
- Batch priority refresh: 4.014 -> 0.614 ms в paired local benchmark, trees идентичны.
- Frozen analysis: 3.619 -> 0.656 s для 50000 transitions, outputs идентичны.

Оставшиеся задачи команды:

1. Areg: подключить replay API к общему learner, сохранить model/target/Adam,
   global decision clock и exploration RNG вместе с replay state при resume.
2. Leon: подключить реальные injected delivery/timeout flags; проверить history,
   remaining-time input, autoreset и shared paired seed masks.
3. Сверить sampling-with-replacement и duplicates с существующим PER команды.
4. Выполнить общий pilot/training budget, frozen analysis реального буфера и
   paired evaluation; не выбирать изменения по held-out результатам.

Команды проверки из этой папки:

```powershell
python -m pytest -q --junitxml=results/new_validation.xml
python -m scripts.synthetic_demo --output results/new_synthetic_segment
python -m scripts.plot_diagnostics --logs results/new_synthetic_segment/logs --frozen-analysis results/new_synthetic_segment/frozen_analysis --output results/new_synthetic_segment/figures
python -m scripts.benchmark_refresh --output results/new_refresh_benchmark.json
python -m scripts.benchmark_frozen --label after --root results/new_frozen_benchmark
```

Replay checkpoint (Python object; включать в доверенный checkpoint общего learner):

```python
checkpoint["replay"] = buffer.state_dict()
# ... resume, при той же версии NumPy/RNG и согласованном learner clock ...
buffer = ReplayBuffer.from_state_dict(checkpoint["replay"])
```

Документы: `RESEARCH_LOG.md`, `docs/INTEGRATION.md`, `docs/LOG_SCHEMA.md`.
Доказательства запусков: `results/improvements/`.
Остановка по заданному лимиту 5 циклов. После предоставления репозитория доступны
environment/outage компоненты команды; общий learner/training/evaluation пока
отсутствует. Актуальные hooks и замечания к wrapper — в корневом INTEGRATION.md,
задачи Areg/Leon — в корневом README. Выводов об эффективности grouped PER нет.
