# CDR Generator — Implementation Plan

**Aligned with:** Requirements v0.7
**Approach:** Инкрементальный — каждая фаза даёт запускаемый результат.

---

## Phase 1 — Skeleton: Config + Assets + Minimal Output

**Цель:** Загрузить конфиг, сгенерировать ассеты, записать пустой CSV.

**Модули:**
- `config/loader.py` — YAML → dataclass, валидация jsonschema, --override
- `config/schema.json` — JSON Schema для конфига
- `assets/generator.py` — генерация cells, NE, subscribers
- `assets/store.py` — сохранение/загрузка/валидация, manifest
- `assets/models.py` — dataclass'ы: Cell, NetworkElement, Subscriber
- `writer/csv_writer.py` — CSV+gzip writer, заголовок по schema, один файл на NE/день
- `cli.py` — entrypoint: generate, generate-assets, validate-config, validate-assets

**Результат запуска:**
```bash
cdrgen generate-assets --config config.yaml --assets-dir ./assets
cdrgen validate-assets --assets-dir ./assets
cdrgen generate --config config.yaml --assets-dir ./assets --output-dir ./output --dry-run
# → output/msc-01/CDR_msc-01_20250101.csv.gz (только заголовок)
```

**Acceptance:**
- Конфиг загружается, невалидный — ошибка с path к проблеме
- Ассеты: cells, NE, subscribers на диске, manifest с хешом конфига
- Валидация ассетов: cell_id ∈ cells, serving_ne ∈ NE, etc.
- Пустые CSV+gzip файлы с правильными именами и заголовками

**Оценка:** ~3 дня

---

## Phase 2 — Single-Threaded Generation: Events + Time-Step Engine

**Цель:** Один процесс генерирует CDR для всех абонентов. Без contact book,
без mobility, без anomalies. Позиция = home_cell. B-party = random.

**Модули:**
- `engine/time_step.py` — итерация по time_step'ам
- `engine/rates.py` — расчёт effective_rate (hourly × dow × special × step)
- `engine/poisson.py` — Poisson sampling
- `generators/voice.py` — MO/MT voice, success/failure, duration
- `generators/data.py` — data session, SGW+PGW, partial records
- `generators/sms.py` — MO/MT SMS
- `models/cdr.py` — CDR record dataclass → CSV row
- `writer/csv_writer.py` — расширить: принимает records, пишет sorted

**Упрощения (будут убраны в следующих фазах):**
- B-party = random subscriber или external number
- Позиция A = home_cell, B = home_cell (нет mobility)
- Без anomalies
- Без vendor extensions
- Без special events
- Один процесс, все абоненты последовательно

**Результат запуска:**
```bash
cdrgen generate --config config.yaml --workers 1
# → output/msc-01/CDR_msc-01_20250101.csv.gz (реальные записи)
# → output/sgw-01/CDR_sgw-01_20250101.csv.gz
# → итоговая статистика в stdout
```

**Acceptance:**
- CDR файлы с реальными записями, отсортированы по event_timestamp
- Количество записей ≈ estimate (±10%)
- Правильные поля: record_type, IMSI, MSISDN, timestamps, duration, cause codes
- SGW+PGW парные (charging_id), MO+MT парные (consolidation_id)
- Partial records для длинных data sessions
- Hourly distribution визуально соответствует весам

**Оценка:** ~5 дней

---

## Phase 3 — Contact Book + Mobility + Extensions

**Цель:** Реалистичные B-party, движение абонентов, vendor-specific поля.

**Модули:**
- `assets/contact_book.py` — генерация (zipf degree, asymmetric, profile bias)
- `assets/external_numbers.py` — пул внешних номеров
- `engine/mobility.py` — чистая функция position(home, work, profile, T, seed)
- `engine/b_party.py` — выбор B: contact book / external / random
- `generators/extensions.py` — vendor extensions → base64 JSON
- Обновить generators/voice.py, sms.py: repeat_call_probability, external_call_ratio

**Результат запуска:**
```bash
cdrgen generate-assets --config config.yaml
# → assets/contact_book.bin (~30 KB при 10K subs)
cdrgen generate --config config.yaml
# → CDR с реалистичными B-party и cell movement
```

**Acceptance:**
- Contact book: zipf distribution, asymmetric, bias проверяем статистически
- B-party: ~60% из contact book, ~15% external, ~25% random
- Позиция: home ночью, work днём, commute в указанные часы
- Handover: first_cell ≠ last_cell в ~5% звонков (по профилю)
- Extensions: base64-encoded JSON, ключи {vendor}.{field}
- Roaming: ~1% для office_worker, ~15% для heavy_traveler

**Оценка:** ~4 дня

---

## Phase 4 — Anomalies + Special Events

**Цель:** Аномалии и спецсобытия делают данные реалистичными для тестирования ETL.

**Модули:**
- `engine/anomalies.py` — pipeline: orphaned → missing → corrupt → timestamps → duplicates
- `engine/special_events.py` — загрузка событий, compound multipliers, disabled cells
- Обновить engine/rates.py: special_event_multiplier
- Обновить engine/time_step.py: disabled cells → overflow / failure

**Результат запуска:**
```bash
cdrgen generate --config config.yaml
# → CDR с аномалиями и спецсобытиями
cdrgen estimate --config config.yaml
# → таблица: CDR/день, объём, anomalies breakdown
```

**Acceptance:**
- Duplicates: ~0.5% с jitter timestamp
- Orphaned: ~2% (MO без MT или наоборот)
- Missing fields: ~1% null в mandatory полях
- Timestamps: ~0.3% future/negative/rollover
- Corrupt: ~0.2% invalid IMSI, impossible cell
- New Year spike: SMS ×20, voice ×5, failure 50%
- Cell outage: subscribers relocated to overflow, failure rate up
- Аномалии применяются генератором до отправки writer'у

**Оценка:** ~3 дня

---

## Phase 5 — Call Forwarding + Edge Cases

**Цель:** Дополнительные сценарии, повышающие реалистичность.

**Модули:**
- Обновить generators/voice.py: call forwarding → 2 CDR (MO + MT redirect)
- `engine/concurrency.py` — 1 voice + 1 data + N sms per step enforcement
- Edge cases: disabled cells without overflow → failure code 38

**Acceptance:**
- Call forwarding: ~3% звонков, MO-CDR с redirecting_number, MT-CDR на C
- Concurrency: не более 1 voice + 1 data per subscriber per step
- Disabled cell без overflow → cause_code=38

**Оценка:** ~2 дня

---

## Phase 6 — Multiprocess: Generator Shards + Writer Workers

**Цель:** Параллелизм для масштаба 10–50M.

**Модули:**
- `engine/orchestrator.py` — разбивка subscribers на shards, запуск generators + writers
- `engine/generator_worker.py` — subscriber shard, отправка CDR writer'ам
- `writer/writer_worker.py` — получение CDR, сортировка, запись файла
- `assets/store.py` — mmap/shared memory для read-only доступа
- Обновить cli.py: --workers, --progress

**Архитектура:**
```
orchestrator
  ├─ generator_worker[0..N-1]  (subscriber shards)
  │     │
  │     ├─→ writer_worker[msc-01, 2025-01-01]
  │     ├─→ writer_worker[msc-01, 2025-01-02]
  │     ├─→ writer_worker[sgw-01, 2025-01-01]
  │     └─→ ...
  │
  └─ progress monitor
```

**Ключевые решения (деталь реализации):**
- Коммуникация gen→writer: на усмотрение реализации (Queue / temp files / pipes)
- Сортировка в writer: на усмотрение (in-memory sort / external sort / presorted input)
- Seed: worker_seed = f(global_seed, shard_id) — детерминированность

**Acceptance:**
- 10K subs, 1 worker = 16 workers: содержимое CDR идентично (diff по полям)
- Файлов всегда NE × дни (не зависит от числа workers)
- Записи отсортированы по event_timestamp
- 10M subs × 1 день < 1 часа (16+ workers)
- Прогресс-бар, итоговая статистика
- Ассеты загружаются shared (не копируются per-worker)

**Оценка:** ~5 дней

---

## Summary

| Phase | Содержание                            | Дней | Накопительный результат                    |
|-------|---------------------------------------|------|--------------------------------------------|
| 1     | Config + Assets + Skeleton            | 3    | Загрузка конфига, ассеты, пустые файлы     |
| 2     | Single-thread generation              | 5    | Реальные CDR, один процесс                 |
| 3     | Contact book + Mobility + Extensions  | 4    | Реалистичные B-party и движение            |
| 4     | Anomalies + Special events            | 3    | Аномалии, спецсобытия                      |
| 5     | Call forwarding + Edge cases          | 2    | Forwarding, concurrency limits             |
| 6     | Multiprocess                          | 5    | Параллельная генерация 10–50M              |
|       | **Итого**                             | **22**|                                            |

**Зависимости:**
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                                  ↘
Phase 2 ─────────────────────────────────────────→ Phase 6
```
Phase 6 может начаться после Phase 2 (минимальная генерация),
но полный acceptance — после Phase 5.
