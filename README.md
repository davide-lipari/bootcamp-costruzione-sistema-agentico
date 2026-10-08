# lipari-bank-ai

L'assistente AI di LipariBank per gli operatori di filiale: un servizio FastAPI su Postgres con pgvector, con un RAG sulle policy della banca filtrato per ruolo, un agente con cinque tool legati all'operatore che chiede, una sosta per l'approvazione di un responsabile prima di una segnalazione alla Compliance sopra soglia, un server MCP, e un harness che misura le risposte.

## Avvio

Servono Docker e [uv](https://docs.astral.sh/uv/). Python 3.12 lo installa uv.

```bash
# 1. Postgres con pgvector, e il database dei test
docker run -d --name lipari-pg -e POSTGRES_USER=lipari -e POSTGRES_PASSWORD=lipari \
  -e POSTGRES_DB=lipari_ai -p 5432:5432 pgvector/pgvector:pg16
docker exec lipari-pg psql -U lipari -d lipari_ai -c "create database lipari_ai_test"

# 2. la configurazione: copia l'esempio e compila i valori che elenca
cp .env.example .env

# 3. dipendenze, schema, dati di sviluppo
uv sync
uv run alembic upgrade head
uv run python -m scripts.seed_users
uv run python -m scripts.seed_accounts

# 4. il servizio, e la sonda
uv run uvicorn src.main:app --port 8000
curl -s localhost:8000/health        # "status":"UP"

# 5. da un secondo terminale, col servizio acceso: i documenti nell'indice
uv run python -m scripts.ingest_docs
```

Il caricamento dei documenti calcola gli embedding con il modello configurato, quindi chiede la chiave API nel `.env`. Le credenziali degli utenti di sviluppo sono in `scripts/seed_users.py`, e valgono solo sulla tua macchina.

## Test e controlli

```bash
uv run pytest -q          # sul database lipari_ai_test
uv run mypy .
uv run ruff check . && uv run ruff format --check .
uv run python -m evals.runner   # l'harness: chiama il modello vero, e costa
```

I test non chiamano nessun modello: usano i finti di `tests/finti.py`.

## Dove sono le cose

| Cartella | Cosa |
|---|---|
| `src/api/` | gli endpoint: chat, consulenza sulle policy, agente, conti, movimenti, autenticazione |
| `src/agents/` | il ciclo dell'agente scritto a mano, i tool, la sosta per l'approvazione, il supervisor, il client MCP |
| `src/services/` | retrieval con i permessi, RAG con le citazioni, ingestione, categorizzazione |
| `liparibank_mcp/` | il server MCP, che verifica l'identità da un token firmato |
| `evals/` | l'harness, i casi in `datasets/`, il cancello con le soglie |
| `data/docs/` | il corpus: le sottocartelle dicono chi può leggere (`compliance_only/`) |

## Misure

Le due esecuzioni di `uv run python -m evals.confronto` del **08/10/2026**, entrambe dopo
`uv run python -m scripts.ingest_docs` (6 documenti in `data/docs/`, 9 casi in
`evals/datasets/agent_traiettorie.jsonl`). «ricerche» = `search_documents` eseguite per caso;
il costo è quello misurato per caso dall'harness.

### Prima — casi nuovi presenti, tetto assente

```text
== ciclo
traiettorie  0.78 (7/9)  soglia 0.80  SOTTO SOGLIA  costo EUR 0.0036
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 3, "stopped_by": "model", "costo_eur": "0.000562"}
    x traj-006: {"violati": ["risposta_contiene"], "tool_chiamati": ["search_documents"], "passi": 3, "stopped_by": "model", "costo_eur": "0.000609"}
  traj-001                 0 ricerche  0.000320 EUR
  traj-002                 0 ricerche  0.000210 EUR
  traj-003                 2 ricerche  0.000562 EUR
  traj-004                 0 ricerche  0.000141 EUR
  traj-005                 0 ricerche  0.000113 EUR
  traj-006                 2 ricerche  0.000609 EUR
  traj-007                 1 ricerche  0.000293 EUR
  traj-008                 1 ricerche  0.000275 EUR
  traj-009                 5 ricerche  0.001095 EUR

== catena
traiettorie  0.22 (2/9)  soglia 0.80  SOTTO SOGLIA  costo EUR 0.0012
    x traj-001: {"violati": ["tool_richiesti"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000101"}
    x traj-002: {"violati": ["tool_richiesti"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000112"}
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000098"}
    x traj-004: {"violati": ["tool_richiesti", "terminazione"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000239"}
    x traj-005: {"violati": ["tool_vietati"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000082"}
    ... e altri 2: traj-006, traj-007
  traj-001                 1 ricerche  0.000101 EUR
  traj-002                 1 ricerche  0.000112 EUR
  traj-003                 1 ricerche  0.000098 EUR
  traj-004                 1 ricerche  0.000239 EUR
  traj-005                 1 ricerche  0.000082 EUR
  traj-006                 1 ricerche  0.000147 EUR
  traj-007                 1 ricerche  0.000136 EUR
  traj-008                 1 ricerche  0.000155 EUR
  traj-009                 1 ricerche  0.000114 EUR

== agente
traiettorie  0.78 (7/9)  soglia 0.80  SOTTO SOGLIA  costo EUR 0.0038
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 2, "stopped_by": "model", "costo_eur": "0.000291"}
    x traj-009: {"violati": ["passi"], "tool_chiamati": ["search_documents"], "passi": 8, "stopped_by": "model", "costo_eur": "0.001791"}
  traj-001                 0 ricerche  0.000319 EUR
  traj-002                 0 ricerche  0.000212 EUR
  traj-003                 1 ricerche  0.000291 EUR
  traj-004                 0 ricerche  0.000151 EUR
  traj-005                 0 ricerche  0.000117 EUR
  traj-006                 1 ricerche  0.000317 EUR
  traj-007                 1 ricerche  0.000345 EUR
  traj-008                 1 ricerche  0.000286 EUR
  traj-009                 7 ricerche  0.001791 EUR
```

### Dopo — tetto attivo (`ToolCallLimitMiddleware` su `search_documents`, `thread_limit=3`, `exit_behavior="continue"`)

```text
== ciclo
traiettorie  0.78 (7/9)  soglia 0.80  SOTTO SOGLIA  costo EUR 0.0024
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 2, "stopped_by": "model", "costo_eur": "0.000262"}
    x traj-008: {"violati": ["risposta_contiene"], "tool_chiamati": ["search_documents"], "passi": 2, "stopped_by": "model", "costo_eur": "0.000279"}
  traj-001                 0 ricerche  0.000315 EUR
  traj-002                 0 ricerche  0.000204 EUR
  traj-003                 1 ricerche  0.000262 EUR
  traj-004                 0 ricerche  0.000170 EUR
  traj-005                 0 ricerche  0.000105 EUR
  traj-006                 1 ricerche  0.000300 EUR
  traj-007                 1 ricerche  0.000278 EUR
  traj-008                 1 ricerche  0.000279 EUR
  traj-009                 2 ricerche  0.000511 EUR

== catena
traiettorie  0.22 (2/9)  soglia 0.80  SOTTO SOGLIA  costo EUR 0.0012
    x traj-001: {"violati": ["tool_richiesti"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000101"}
    x traj-002: {"violati": ["tool_richiesti"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000112"}
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000098"}
    x traj-004: {"violati": ["tool_richiesti", "terminazione"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000275"}
    x traj-005: {"violati": ["tool_vietati"], "tool_chiamati": ["search_documents"], "passi": 1, "stopped_by": "model", "costo_eur": "0.000082"}
    ... e altri 2: traj-006, traj-007
  traj-001                 1 ricerche  0.000101 EUR
  traj-002                 1 ricerche  0.000112 EUR
  traj-003                 1 ricerche  0.000098 EUR
  traj-004                 1 ricerche  0.000275 EUR
  traj-005                 1 ricerche  0.000082 EUR
  traj-006                 1 ricerche  0.000133 EUR
  traj-007                 1 ricerche  0.000136 EUR
  traj-008                 1 ricerche  0.000158 EUR
  traj-009                 1 ricerche  0.000114 EUR

== agente
traiettorie  0.89 (8/9)  soglia 0.80  ok            costo EUR 0.0027
    x traj-003: {"violati": ["citazione"], "tool_chiamati": ["search_documents"], "passi": 2, "stopped_by": "model", "costo_eur": "0.000291"}
  traj-001                 0 ricerche  0.000309 EUR
  traj-002                 0 ricerche  0.000199 EUR
  traj-003                 1 ricerche  0.000291 EUR
  traj-004                 0 ricerche  0.000147 EUR
  traj-005                 0 ricerche  0.000117 EUR
  traj-006                 1 ricerche  0.000334 EUR
  traj-007                 1 ricerche  0.000326 EUR
  traj-008                 1 ricerche  0.000291 EUR
  traj-009                 3 ricerche  0.000680 EUR
```

### Delta (08/10/2026, «prima» → «dopo»)

| gruppo (casi) | catena: ricerche | catena: EUR | agente: ricerche | agente: EUR |
|---|---|---|---|---|
| un passo (traj-001…005) | 5, uno per caso, fisso | 0.000632 → 0.000668 | 1 → 1 | 0.001090 → 0.001063 |
| multi-hop (traj-006…008) | 3, uno per caso, fisso | 0.000438 → 0.000427 | 3 → 3 | 0.000948 → 0.000951 |
| senza risposta (traj-009) | 1 → 1 | 0.000114 → 0.000114 | **7 → 3** | 0.001791 → 0.000680 |
| totale ricerche | 9 → 9 | | **11 → 7** | |

- **Casi a un passo: la catena resta più economica, e il numero lo mostra**: 0.000632 EUR
  (prima) contro 0.001090 dell'agente sui cinque casi, pur con una ricerca fissa per caso
  (l'agente ne fa una sola in tutto il gruppo). Anche «dopo» il divario c'è: 0.000668 contro
  0.001063.
- **Multi-hop**: l'agente supera 3 casi su 3 in entrambe le esecuzioni, la catena 1 su 3
  (solo traj-008); i costi per caso non cambiano con il tetto (1 ricerca a caso in tutte e due).
- **Senza risposta (traj-009)**: è l'unico punto dove il tetto si vede. L'agente prima 7 ricerche
  e 8 passi (fallito sul controllo `passi`), dopo 3 ricerche sotto il tetto, terminazione
  `model`: il caso passa e scende da 0.001791 a 0.000680 EUR.
- **Punteggi**: catena 0.22 → 0.22 (sotto soglia 0.80 in entrambe), agente 0.78 → 0.89 (sopra
  soglia da «dopo»), ciclo 0.78 → 0.78.
- `catena` e `ciclo` non passano dal middleware: fra le due esecuzioni i loro numeri cambiano
  solo per il modello non deterministico (la ricerca della catena resta una, il costo dei casi
  a un passo oscilla di poche unità alla quinta cifra).
- Un fallimento che non c'entra col tetto: traj-003 non supera `citazione` in nessuna delle
  due esecuzioni.

