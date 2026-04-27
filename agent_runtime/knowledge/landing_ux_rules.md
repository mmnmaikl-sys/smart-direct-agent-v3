# Landing UX Rules — БФЛ-лендинги 2026

**Owner:** marketer-no1 + Михаил Мяклов
**Source:** Direct day-1 incident 26-27.04.2026 (CR=0% при 39 кликах)
**Severity:** P0 — нарушение блокирует mobile-конверсию полностью

---

## Rule 1: Exit-modal ТОЛЬКО на desktop

### Проблема

History pushState + popstate listener БЕЗ device-guard ломает mobile-воронку:

- iOS swipe-back triggers popstate → exit-modal `z-index:200` overrides quiz radio buttons
- Hash-anchor navigation (`<a href="#quiz">`) triggers popstate → same effect
- Playwright iPhone 13 reproduce: 5 retry-clicks on radio button — все intercepted

### Симптомы

- Mobile bounce ≥ 60% с avg session 15 sec
- 0 quiz_completed на mobile при ≥10 visits
- Direct-агент видит CR=0 → STOP кампаний

### Правило

**Exit-modal popstate listener** должен быть обёрнут в:

```javascript
if (window.matchMedia('(min-width:768px)').matches) {
  history.replaceState({page:'lp'}, '');
  history.pushState({page:'lp2'}, '');
  window.addEventListener('popstate', function(){
    if (!exitShown) { history.pushState({page:'lp2'}, ''); showExit(); }
  });
}
```

### Anti-pattern (запрещено)

```javascript
// ❌ Без mobile-guard — ломает iOS swipe-back
history.pushState({page:'lp'}, '');
window.addEventListener('popstate', () => showExit());
```

### Verification

- Playwright iPhone 13 + Pixel 5: квиз должен пройти все 3 шага без модалки
- Метрика: mobile bounce должен быть < 60% после фикса
- Direct CR ≥ 3% за 100+ кликов после resume кампаний

---

## Rule 2: Mouseleave exit-intent — тоже только desktop

`mouseleave` triggered by mouse exiting top of viewport — на mobile **отсутствует** (нет курсора). Listener бесполезен:

```javascript
if (window.matchMedia('(min-width:768px)').matches) {
  document.addEventListener('mouseleave', function(e){
    if (e.clientY < 10) showExit();
  });
}
```

Уже соблюдается в `/lp/v2-outcome/` v1. Сохранить.

---

## Rule 3: ФЗ-332 warning ОБЯЗАТЕЛЕН на каждом БФЛ-лендинге

Footer / sticky-bottom warning: «Услуга связана с банкротством, имеются риски».

Рекомендуемая площадь: ≥ 7% screen height. Видимый цвет (контраст AA).

См. также `legal-compliance-wave1.md`.

---

## Rule 4: Quiz steps ≤ 6 для холодного трафика

- Direct cold traffic: 3-6 шагов оптимально
- Phone capture step — финальный
- Form fields max 2 per step

См. также `channel_mastery/landing.md` (G1: maketornado A-grade case CR 1.5%→7.5% после refactor 12→5 step).

---

## Rule 5: Inline submit-handler должен передавать UTM в Bitrix

Каждое submit-event должно передать в Bitrix:
- UTM_SOURCE, UTM_MEDIUM, UTM_CAMPAIGN, UTM_CONTENT, UTM_TERM
- yclid (Yandex click id)
- referrer
- ab_variant (если A/B test)

Verify через Playwright probe: submit → Bitrix lead с правильным UTM_CAMPAIGN ≤ 30 сек.

---

## Rule 6: Метрика goal-events trigger при каждом шаге

Минимум:
- `quiz_step_2`, `quiz_step_3`, ..., `quiz_step_N` per шаг
- `quiz_completed` после финального submit
- `lead_form_submitted` если есть отдельная форма
- `phone_clicked` на tel: ссылках

Composite goal: реакция при ANY of (quiz / form / call) — `id=549152162`.

---

## Когда применять эти правила

- При создании любого нового лендинга на 24bankrotsttvo.ru/lp/
- При review существующих лендингов (cron landing_critic.py)
- Перед `ftp_deploy.py` в production
- После любого AI-генерации HTML (если landing-builder появится)

## Owner / change log

- 2026-04-27 (v1): Rule 1 (exit-modal mobile-guard) после Direct day-1 incident, P0 fix deployed via FTP + commit 19f77e6 в mmnmaikl-sys/24bankrotsttvo-lp
