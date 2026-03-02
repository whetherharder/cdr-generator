# Performance Optimization: 50k CDR/sec on Single Core

## Overview

Достичь 50,000 CDR/sec на одном ядре CPU (цель `test_single_core_throughput_minimum`).
Текущий базовый показатель: ~1,500 CDR/sec (нужно ~33x).

**Ключевой инсайт:** 98% итераций внутреннего цикла `(subscriber × timestep)` дают 0 событий —
Python платит накладные расходы за 144,000 пустых итераций, из которых реальные события
в ~2,941 случаях. Устранение пустых итераций через батчевый Poisson даёт потенциал 30-50x.

**Целевые объёмы:** миллионы абонентов за месяцы симулированного времени.

## Context (from discovery)

- Файлы горячего пути: `engine/runner.py`, `generators/voice.py`, `generators/sms.py`,
  `generators/data.py`, `engine/b_party.py`, `engine/mobility.py`
- Уже сделанные оптимизации (commit 733dac9): `_ProfileCache.__slots__`,
  предвычисленные weight_sum, aliasing dict'ов, кэш timedelta
- Бенчмарк-инфраструктура: `tests/benchmark_generation.py` (поддерживает `--profile`)
- Тест цели: `tests/test_performance.py::test_single_core_throughput_minimum`
- 434 теста должны оставаться зелёными на всём протяжении работы

## Development Approach

- **Testing approach**: Regular (код → тесты)
- Замерять производительность до и после каждого изменения
- Проверять функциональность после каждой задачи: все 434 теста должны проходить
- Любые библиотеки допустимы если дают реальный прирост
- Менять только то, что нужно — не рефакторить окружающий код

## Progress Tracking

- Отмечать выполненные пункты `[x]` сразу после завершения
- Добавлять новые задачи с префиксом ➕
- Документировать блокеры с префиксом ⚠️

## Implementation Steps

---

### Task 0: Профилирование — baseline замер

- [x] запустить `python tests/benchmark_generation.py --profile --subscribers 200 --hours 24`
- [x] зафиксировать wall time, CDR/sec, top-10 функций по cumtime в плане (ниже)
- [x] запустить `pytest tests/ -q --tb=short` — убедиться что 434 теста зелёные
- [x] зафиксировать baseline: 355 CDR/sec (заполнить после измерения)

**Baseline profiling results:**
```
CDR/sec: 355  (200 subs, 24h, 5828 records in 16.4s)
Wall time: 16.419s
Total records: 5,828 (voice=854, sms=0, data=4974)

Top-5 by cumtime (200 subs × 1440 steps):
  1. mobility.resolve_position:   3.087s (288,000 calls — every sub every step!)
  2. poisson.sample_count:        2.949s (864,000 calls — 200 subs × 1440 × 3 types)
  3. dict.get (inline):           2.049s (1,607,513 calls)
  4. rates.effective_rate:        1.161s (864,000 calls — same as poisson)
  5. mobility._maybe_handover:    1.037s (288,000 calls)

Key insight confirmed: 864,000 poisson calls for only 5,828 events (~0.67% non-zero).
  mobility called 288,000× even though only ~2,941 events needed it.
  Both will be eliminated by vectorized Poisson batch (Task 1).

pytest results: 435 passed, 1 failed (test_single_core_throughput_minimum — target),
  1 skipped. All functional tests green.
Performance test baseline: 1,528 CDR/sec (test config: 100 subs, 1h).
```

---

### Task 1: Векторизованный Poisson — убрать 98% пустых итераций

Ключевая идея: вместо вызова `rng.poisson(rate)` в цикле per-(sub, step) —
вычислить все счётчики за одну операцию `rng.poisson(rate_matrix)`.

Место: `engine/runner.py` (основной цикл).

**Структура изменения:**

```python
# Было (внутри двойного цикла):
for step_t in time_steps:                    # 1440 итераций
    for sub, pc in sub_profile_pairs:        # 100 итераций
        n_voice = poisson.sample_count(voice_rate, rng)  # отдельный вызов
        n_sms   = poisson.sample_count(sms_rate,   rng)
        n_data  = poisson.sample_count(data_rate,  rng)
        if n_voice > 0: ...                  # 98% раз — мимо

# Стало:
# Per-hour (24 группы по 60 steps):
#   rate_matrix[sub_idx, event_type] — статичен внутри часа
#   counts = rng.poisson(rate_matrix, size=(60, n_subs, 3))  # всё за один вызов
# Итерировать только np.argwhere(counts > 0)
```

- [x] добавить helper `_build_rate_matrix(sub_profile_pairs, hour, dow, step_seconds, special_mult) -> np.ndarray` shape `(n_subs, 3)` в `engine/runner.py`
- [x] реструктурировать основной цикл: группировать шаги по часам (1440 шагов → 24 × 60)
- [x] для каждой часовой группы: `counts = rng.poisson(rate_matrix[np.newaxis], size=(60, n_subs, 3))`
- [x] использовать `np.argwhere(counts > 0)` → список `(step_offset, sub_idx, event_type, count)`
- [x] внутренний цикл по non-zero событиям (только реальные события, не пустые итерации)
- [x] убедиться что mobility вызывается только для non-zero (step, sub) пар
- [x] запустить `pytest tests/ -q --tb=short` — все тесты зелёные
- [x] замерить CDR/sec до и после → зафиксировать в плане

**After Task 1:** 12,710 CDR/sec (200 subs, 24h) — baseline was 355 CDR/sec (~36x improvement)
Performance test (100 subs, 1h): 13,664 CDR/sec — baseline was 1,528 CDR/sec (~9x)

---

### Task 2: Батчевые буферы случайных чисел

Ключевая идея: вместо одиночных вызовов `rng.lognormal()`, `rng.integers()` per-event
— заполнять deque пачками по 2048 элементов.

Место: `engine/runner.py` (инициализация до цикла) + generators.

- [x] реализовать класс `_RngBuffer` (или модуль `engine/rng_buffer.py`) с методами:
  - `get_uuid() -> str` — берёт из pre-filled буфера `os.urandom(16).hex()` пачками
  - `get_lognormal(mu, sigma) -> float` — буфер per-(mu, sigma) пары
  - `get_uniform(lo, hi) -> float`
  - `get_choice(n, p) -> int` — `rng.choice(n, p=p, size=BATCH)` пачкой
- [x] батчевые UUID: `raw = rng.bytes(16 * BATCH)` → нарезать на 16-байтовые куски → `.hex()`
  (или `os.urandom` если не нужен детерминированный seed; для детерминированности — через rng)
- [x] заменить вызовы UUID в `generators/voice.py` и `generators/sms.py` на буфер
- [x] заменить вызовы `_sample_distribution(lognormal)` на буфер в `generators/voice.py`, `sms.py`, `data.py`
- [x] заменить `_weighted_choice` на буфер `rng.choice(n, p=p, size=BATCH)` в generators
- [x] запустить `pytest tests/ -q --tb=short` — все тесты зелёные
- [x] замерить CDR/sec → зафиксировать в плане

**After Task 2:** 16,868 CDR/sec (200 subs, 24h) — was 12,710 CDR/sec (~33% improvement)
Performance test (100 subs, 1h): ~19,625 CDR/sec — was 13,664 CDR/sec (~44% improvement)

---

### Task 3: Лёгкие записи — убрать overhead Python объектов

Ключевая идея: `@dataclass CDRRecord` создаёт тяжёлый объект с 26 полями.
Замена на `__slots__` dataclass или namedtuple снижает аллокацию.

Место: `models/cdr.py` + `writer/csv_writer.py`.

- [x] добавить `__slots__` к `CDRRecord` dataclass (или заменить на `typing.NamedTuple`)
- [x] убедиться что все поля Optional с default=None сохраняются
- [x] проверить что `writer/csv_writer.py` корректно работает с новой структурой
- [x] запустить `pytest tests/ -q --tb=short` — все тесты зелёные
- [x] замерить CDR/sec → зафиксировать в плане

**After Task 3:** ~17,000 CDR/sec (was 16,868 — improvement within noise; CDRRecord allocation not primary bottleneck)

---

### Task 4: Datetime оптимизация — минимизировать арифметику

Ключевая идея: datetime-объекты в Python дорогие. Хранить offset в секундах (float),
прибавлять к base_datetime только при сериализации.

Место: `engine/runner.py` (передача step_time в generators) + generators.

- [x] профилировать долю времени на `datetime.__add__`, `timedelta`, `.strftime` в текущем коде
- [x] если значимо (>5% от cumtime): изменить generators на приём `base_ts: datetime, offset_sec: float`
  и вычисление итогового timestamp только в `writer/csv_writer.py` через одно сложение
- [x] предвычислить `ISO_PREFIX = step_time.strftime("%Y-%m-%dT%H:%M")` per-step, добавлять только секунды и миллисекунды
- [x] запустить `pytest tests/ -q --tb=short` — все тесты зелёные
- [x] замерить CDR/sec → зафиксировать в плане

**Profiling results (after Tasks 1-3):**
```
strftime: 17,947 calls, 0.073s cumtime = 2.7% of total — below 5% threshold.
Generator change (base_ts + offset_sec) skipped per plan rule.

Actual bottleneck discovered: to_csv_row + _format_value = 1.340s cumtime (50% of total).
Root cause: generic getattr loop (156k calls, 0.193s) + isinstance dispatch (186k calls, 0.230s).

Optimization applied: rewrite to_csv_row with explicit attribute access + type-specific
formatting, eliminating getattr/isinstance overhead entirely.
to_csv_row cumtime: 1.340s → 0.164s (8x speedup in profiling).
```

**After Task 4:** ~20,000 CDR/sec (100 subs, 24h) — was ~17,000 CDR/sec (~18% improvement)

---

### Task 5: Дополнительные micro-оптимизации по профайлеру

После Tasks 1-4 запустить профайлер повторно и устранить оставшиеся hotspot'ы.
Типичные кандидаты (реализовать только те, что реально видны в профиле):

- [x] запустить профайлер повторно, зафиксировать новый top-5
- [x] contact_book lookup: кэшировать `contact_book.get_contacts(imsi)` per-subscriber на старте
  (список контактов стабилен в течение генерации)
- [x] vendor extensions: если `_apply_vendor_extensions` в top-5 — batch encode per-step
- [x] anomaly pipeline: профилировать overhead batch-apply на write фазе
- [x] запустить `pytest tests/ -q --tb=short` — все тесты зелёные
- [x] замерить CDR/sec → зафиксировать в плане

**Profiling results (after Tasks 1-4, before Task 5):**
```
New top-5 by tottime:
  1. dict.get:                       0.169s (134,280 calls) — static config lookups in generators
  2. _weighted_sample_without_repl:  0.136s (200 calls)    — startup: build_contact_book
  3. list.append:                    0.109s (87,664 calls)  — CDR accumulation
  4. build_contact_book:             0.103s (1 call)        — one-time startup cost
  5. run_generation (self):          0.102s (1 call)        — inner loop overhead

Vendor extensions: NOT in top-5 — skipped per plan.
Anomaly pipeline: NOT in top-5 — skipped per plan.
```

**Task 5 optimizations applied:**
- `_DataCfgCache` in data.py: pre-computes apn_probs, qci_probs, term_probs, dur/vol params
  → bypasses all dict.get in generate_data_cdr hot path
  → generate_data_cdr cumtime: 0.460s → 0.173s (63% reduction)
- `_VoiceCfgCache` in voice.py: pre-computes success_rate, cause probs, dur params, jitter_hi
  → eliminates voice dict.get calls in _generate_failed/successful/forwarded_call
- `sub_contacts[sub_idx]` in runner.py: pre-built contact list per subscriber
  → passes a_party_contacts to select_b_party, avoids contact_book.get() per call
- Pre-extracted `_b_ext_ratio`, `_b_contact_threshold` as floats
  → eliminates 2 config.get() calls per b_party selection
- dict.get calls: 134,280 → 78,786 (41% reduction)

**After Task 5:** ~23,000 CDR/sec (100 subs, 24h) — was ~20,000 CDR/sec (~15% improvement)

---

### Task 6: Верификация результата и финальная очистка

- [ ] запустить `pytest tests/test_performance.py -v` — `test_single_core_throughput_minimum` должен пройти
- [ ] запустить `pytest tests/ -q --tb=short` — все 434 теста зелёные
- [ ] запустить `ruff check cdr_generator tests` — 0 ошибок
- [ ] запустить бенчмарк на нескольких конфигурациях: 100 subs, 500 subs, 1000 subs — убедиться в линейном масштабировании
- [ ] зафиксировать финальные числа в этом плане:

**Финальный результат:**
```
До оптимизации:    ~1,500 CDR/sec
После оптимизации: _______ CDR/sec
Прирост:           _______x
```

---

### Task N-1: Коммит и PR

- [ ] `git checkout develop && git pull`
- [ ] `git checkout -b feature/performance-50k-optimization`
- [ ] `git add` конкретные файлы (не `git add .`)
- [ ] `pytest tests/ -q --tb=short` — финальная проверка перед коммитом
- [ ] `git commit -m "perf: vectorized Poisson + batch RNG — 50k CDR/sec single core"`
- [ ] `gh pr create --base develop`

---

## Technical Details

### Rate matrix structure

```python
# shape: (n_subs, 3) — [voice_rate, sms_rate, data_rate] per subscriber
# rates computed once per hour (changes only when hour/dow changes)
rate_matrix = np.array([
    [pc.voice_lambda * hourly_w[h] / voice_wsum * dow_v * step_factor,
     pc.sms_lambda  * hourly_w[h] / sms_wsum   * dow_s * step_factor,
     pc.data_lambda * hourly_w[h] / data_wsum   * dow_d * step_factor]
    for (sub, pc) in sub_profile_pairs
])  # shape: (n_subs, 3)

# Batch sample for all 60 steps of the hour:
counts = rng.poisson(rate_matrix, size=(60, n_subs, 3))  # shape: (60, n_subs, 3)

# Non-zero events only:
events = np.argwhere(counts > 0)  # shape: (K, 3) — (step_offset, sub_idx, event_type)
```

### Batch UUID generation

```python
BATCH = 2048

def _fill_uuid_buf(rng: np.random.Generator) -> list[str]:
    raw = rng.bytes(16 * BATCH)   # deterministic, one call
    return [raw[i*16:(i+1)*16].hex() for i in range(BATCH)]
```

### Determinism

Порядок обращения к RNG должен сохраняться для детерминированного seed.
Батчевый Poisson изменяет порядок вызовов → детерминизм в рамках
одного и того же run будет сохранён, но результат при одном seed
**изменится** по сравнению со старой реализацией. Это допустимо,
если тест `test_deterministic_output` проверяет только внутреннюю
консистентность (тот же seed → тот же результат), а не конкретные значения.

## Post-Completion

**Замер на целевом масштабе:**
- После merge прогнать бенчмарк на 10K subscribers / 30 days для оценки
  реального времени генерации production-объёмов

**Multiprocess (Phase 6):**
- После этой оптимизации пересчитать ожидаемый throughput с N workers
  (N × 50k CDR/sec = горизонтальное масштабирование)
