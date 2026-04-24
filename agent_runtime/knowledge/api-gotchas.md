# Яндекс.Директ API v5 — gotchas and conventions

Cumulative hit-list, собранный из production-опыта SDA v2.1 и predecessor'ов.
Обновляется при каждой новой API-ошибке (append-only; не удалять закрытые,
пометить ✅ resolved с датой).

## Region IDs — через API, не hardcode

**Ошибка:** в коде `region_ids = [11111, 11119]` — при смене справочника Яндекса hardcode ломается.
**Fix:** запрос `dictionaries.get({"DictionaryNames": ["GeoRegions"]})` → выборка по `Name` русское → `GeoRegionId`.
**Проверенные значения (2026-03):** Башкортостан=11111, Татарстан=11119, Удмуртия=11108. Кэшировать в `sda_state` на 30 дней.

## AdGroup → немедленно suspend AT keys

**Ошибка:** `AdGroups.add` создаёт группу с включённым Autotargeting по умолчанию. Если не выключить — первые часы агент тратит на «широкие» ключи, CPA раздувается.
**Fix:** сразу после создания: `KeywordsAndRetargetings.suspend` по всем Autotargeting-категориям (ALL/MENTIONS/ALTERNATIVES).
**Check:** `keywords.get(...) → KeywordsAndRetargetings[].StateEnum` — все AT должны быть `SUSPENDED`.

## Live-статус — только через API, не кэшировать

**Ошибка:** кэширование статусов кампаний в PG — «живой» prod изменится через Yandex UI, агент не узнает 30+ минут.
**Fix:** statusы (`Running/Suspended/Archived`) читать `campaigns.get({"FieldNames": ["State","Status"]})` перед каждым SET. Кэш — только для неизменных полей (Name, Type).

## BiddingStrategy — формат разный per-type

**Ошибка:** унифицированный JSON для всех стратегий не работает — у `AVG_CPA` поле `AverageCpa`, у `AVG_CPC` поле `AverageCpc`, у `MAXIMUM_CLICKS` — нет target поля вообще.
**Fix:** switch по `Search.BiddingStrategyType`:
- `HIGHEST_POSITION` → `{ Search: { BiddingStrategyType: "HIGHEST_POSITION" } }`
- `AVERAGE_CPA` → `{ Search: { BiddingStrategyType: "AVERAGE_CPA", AverageCpa: { AverageCpa: 80000000 }}}` (в копейках! × 1_000_000)
- `AVERAGE_CPC` → `{ Search: { BiddingStrategyType: "AVERAGE_CPC", AverageCpc: {...}, WeeklySpendLimit: ... }}`

## Деньги — всегда в микро-валюте (копейки × 10^6)

**Ошибка:** `AverageCpa: 1000` = агент платит 1 копейку, кампания стопится за показами.
**Fix:** `AverageCpa = rubles * 1_000_000`. Проверка: чтение возвращает то же число.

## Campaigns.update — идемпотентен; SetBids — нет

**Ошибка:** retry SetBids при network error удвоил ставки в 2 раза.
**Fix:**
- `Campaigns.update` — безопасно ретраить; server merge'ит на Id.
- `SetBids` — идемпотентен только в рамках одного `Bid` value; если retry с новым value — применится последнее.
- Pattern: GET → verify → DECISION → SET → GET-verify. Не retry SetBids без проверки текущего bid через GET.

## IncomeGrade — UI-only для ЕПК

**Ошибка:** в API для Единой Перфоманс Кампании (ЕПК) параметр IncomeGrade недоступен, только через UI.
**Fix:** Для ЕПК не пытаться настраивать через `Campaigns.update`. Агент SUSPEND'ит кампанию, бьёт в Telegram `ASK`, владелец меняет в UI.

## Reports API — lag 5-30 минут

**Ошибка:** вызываем reports за последние 15 минут — пусто, «нет данных».
**Fix:** окно не меньше 1 часа для надёжности. Для real-time мониторинга bid/spend — `campaigns.get` со StatsAll fields (быстрее, меньше данных).

## Schedule — коэффициенты кратны 10

**Ошибка:** `BidModifier.HourlyBidMultipliers: 95` → API возвращает 400 "multiplier not multiple of 10".
**Fix:** только 0/10/20/.../200 (процентные коэффициенты, кратные 10). Округлять любой расчёт: `round(value/10)*10`.

## Demographics — массив объектов с полем `Value`

**Ошибка:** передали `Demographics: "MALE_18_24_0"` → 400.
**Fix:** формат:
```json
"Demographics": [
    {"Gender": "MALE", "Age": "AGE_0_17", "BidModifier": 0},
    {"Gender": "FEMALE", "Age": "AGE_18_24", "BidModifier": 0}
]
```

## Retargeting — через `RetargetingListIds`, не условия

**Ошибка:** пытались создать ретаргет с условиями в BidModifier.
**Fix:** ретаргет — отдельная сущность `retargetingLists.add`; на кампанию применяется через `bids.set` с `BidModifier` на уже созданный список.

## Mobile modifier — только на Поиске, не на РСЯ

**Ошибка:** `MobileBidModifier: -30` на РСЯ-кампанию → 400 "MobileBidModifier not applicable".
**Fix:** проверить `Type`: `TEXT_CAMPAIGN` (Поиск) — можно; `CONTENT_CAMPAIGN` (РСЯ) — нельзя.

## Не использовать PriorityGoal до 3+ конверсий/мес

**Ошибка:** добавили `PriorityGoal` при 0-1 конверсиях — автостратегия «прилипает» к случайным сигналам, CPA взлетает.
**Fix:** wait-gate: добавлять PriorityGoal (offline_deal) только после ≥3 deals за последний месяц.

## DirectAPI() / Config() helpers — из `sdk/bankruptcy`

Legacy SDA v2.1 использовал собственный `sdk/bankruptcy/direct.py`. В SDA v3 — прямые httpx-вызовы через `agent_runtime.tools.direct_api` (Task 7). Не импортировать legacy.
