# Guida completa a LipariBank AI

Guida di apprendimento del progetto: come si avvia, come si usa, cosa fa ogni API, cosa copre
ogni test. Tutti i comandi sono eseguibili dalla radice del progetto con [uv](https://docs.astral.sh/uv/).

> Contenuto:
> 1. [Che cos'è il progetto](#1-che-cosè-il-progetto)
> 2. [Avvio passo passo](#2-avvio-passo-passo)
> 3. [Mappa del codice](#3-mappa-del-codice)
> 4. [Comandi rapidi](#4-comandi-rapidi)
> 5. [Le API REST — elenco completo](#5-le-api-rest--elenco-completo)
> 6. [Autenticazione e ruoli](#6-autenticazione-e-ruoli)
> 7. [L'agente: tool, approvazione, supervisor, grafo](#7-lagente-tool-approvazione-supervisor-grafo)
> 8. [Il server MCP](#8-il-server-mcp)
> 9. [L'agente A2A](#9-lagente-a2a)
> 10. [I test — infrastruttura ed elenco completo](#10-i-test--infrastruttura-ed-elenco-completo)
> 11. [Le eval](#11-le-eval)
> 12. [Il database](#12-il-database)
> 13. [Catalogo degli errori](#13-catalogo-degli-errori)

---

## 1. Che cos'è il progetto

**LipariBank AI** è un assistente AI interno per gli operatori di filiale di una banca. È un
servizio **FastAPI** su **Postgres + pgvector** che compila:

- un **RAG** sulle policy della banca, **filtrato per ruolo** (l'operatore non legge i documenti
  della Compliance);
- un **agente con cinque tool** legati all'operatore che chiede (conti, saldi, movimenti,
  ricerca policy, apertura segnalazioni);
- una **sosta per l'approvazione** di un responsabile prima di una segnalazione sopra soglia
  (5000 €);
- un **server MCP** (Model Context Protocol) che pubblica gli stessi tool dietro token;
- un **agente A2A** (Agent-to-Agent) che espone un'istruttoria come servizio;
- un **grafo LangGraph** che è la quarta implementazione dello stesso comportamento dell'agente;
- un **harness di eval** che misura le risposte con soglie e gate (exit code per la CI).

Quattro implementazioni dello stesso comportamento, misurate dalle eval:
`ciclo` (scritto a mano), `catena` (una sola chiamata), `agente` (prebuilt LangChain), `grafo`
(LangGraph col supervisor).

---

## 2. Avvio passo passo

Servono **Docker** e **uv**. Python 3.12 lo installa uv.

### 2.1 Il database

```bash
docker run -d --name lipari-pg -e POSTGRES_USER=lipari -e POSTGRES_PASSWORD=lipari \
  -e POSTGRES_DB=lipari_ai -p 5432:5432 pgvector/pgvector:pg16
docker exec lipari-pg psql -U lipari -d lipari_ai -c "create database lipari_ai_test"
```

Il secondo comando crea `lipari_ai_test`: **è obbligatorio anche per i test**, ogni run di
pytest applica le migration su quel database.

### 2.2 La configurazione

```bash
cp .env.example .env
```

Compila nel `.env` almeno:

| Variabile | Cos'è |
|---|---|
| `DATABASE_URL` | es. `postgresql+asyncpg://lipari:lipari@localhost:5432/lipari_ai` |
| `JWT_SECRET` | almeno 32 caratteri: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `OPENAI_API_KEY` | chiave vera (serve per ingestione documenti e modello) |
| `SEED_PASSWORD` | password degli utenti di sviluppo, minimo 12 caratteri |

### 2.3 Dipendenze, schema, dati

```bash
uv sync                                # dipendenze (Python 3.12 incluso)
uv run alembic upgrade head            # crea lo schema
uv run python -m scripts.seed_users    # 3 utenti di sviluppo
uv run python -m scripts.seed_accounts # clienti, conti, movimenti
```

Utenti creati dal seed:

| Username | Ruolo | Servono a |
|---|---|---|
| `mbianchi` | `operator` | vedere il proprio portafoglio |
| `grossi` | `compliance_lead` | leggere documenti riservati, ingerire, approvare |
| `lverdi` | `risk_lead` | leggere risk_only, vedere i report costi, approvare |

La password non sta nel codice: sta in `SEED_PASSWORD`. Per riscriverla sugli utenti esistenti:
`uv run python -m scripts.seed_users --ruota`.

### 2.4 Il servizio

```bash
uv run uvicorn src.main:app --port 8000
curl -s localhost:8000/health          # deve rispondere "status":"UP"
```

La documentazione interattiva è su `http://localhost:8000/docs` (Swagger).

### 2.5 I documenti nell'indice RAG

Da un secondo terminale, **col servizio acceso** (l'ingestione passa dall'API HTTP):

```bash
uv run python -m scripts.ingest_docs
```

Legge ogni `data/docs/**/*.md`: la **sottocartella diventa il livello di visibilità**
(`compliance_only/aml_….md` → `compliance_only`). Serve la chiave API perché calcola gli
embedding.

---

## 3. Mappa del codice

| Cartella | Contenuto |
|---|---|
| `src/api/` | gli endpoint FastAPI (chat, advice, agente, conti, auth, admin) |
| `src/agents/` | il ciclo dell'agente a mano, i 5 tool, l'approvazione, il supervisor, i client MCP |
| `src/lg/` | il grafo LangGraph: stato, tabella PERCORSI, checkpointer Postgres, memoria, delega |
| `src/services/` | retrieval con permessi, RAG con citazioni, ingestione, categorizzazione, chat |
| `src/auth/` | JWT, dipendenze, ACL dei ruoli, token MCP |
| `src/db/` | modelli SQLAlchemy e repository |
| `src/llm/` | provider OpenAI/Anthropic, factory, embedder, prompt |
| `scripts/` | seed, ingestione, disegno del grafo |
| `liparibank_mcp/` | il server MCP (stdio e HTTP) |
| `liparibank_a2a/` | l'agente A2A (agent card + JSON-RPC) |
| `evals/` | harness, confronto tra implementazioni, superficie, dataset |
| `registry/` | la voce `server.json` del server MCP + schema |
| `data/docs/` | il corpus documentale (sottocartella = visibilità) |
| `tests/` | la suite: finti, banchi, 20 file di test |
| `docs/` | `grafo.md`, generato dal codice |
| `spiegazione_grafo.md` | spiegazione passo passo del grafo |

---

## 4. Comandi rapidi

```bash
# --- avvio ---
uv run alembic upgrade head
uv run python -m scripts.seed_users
uv run python -m scripts.seed_accounts
uv run uvicorn src.main:app --port 8000
uv run python -m scripts.ingest_docs            # servizio acceso

# --- server pubblicati ---
uv run python -m liparibank_mcp.server          # MCP stdio (LIPARI_TOKEN nell'ambiente)
uv run python -m liparibank_mcp.pubblico        # MCP HTTP su 127.0.0.1:8200 (endpoint /mcp)
uv run python -m liparibank_a2a.agente          # A2A su 127.0.0.1:8300
uv run python -m src.auth.mcp_tokens assistente-filiale   # stampa il segreto del client

# --- test e controlli ---
uv run pytest -q                                # tutta la suite (serve Postgres lipari_ai_test)
uv run pytest tests/test_g2_langgraph.py -q     # un file
uv run pytest tests/unit -q                     # una cartella
uv run pytest tests/test_g2_langgraph.py -q -k sosta   # per nome
uv run pytest --cov=src --cov-report=term-missing -q   # copertura
uv run mypy .
uv run ruff check . && uv run ruff format --check .

# --- eval (chiamano il modello vero: costa) ---
uv run python -m evals.runner                   # gate con soglie (exit 0/1)
uv run python -m evals.confronto                # le 4 implementazioni sugli stessi casi
uv run python -m evals.superficie               # quali tool vede ogni client×ruolo
uv run python -m evals.durata                   # conversazione di 50 turni
uv run python -m evals.ingest_fixtures           # ingestione diretta, senza API (per la CI)

# --- documenti generati ---
uv run python -m scripts.disegna_grafo          # riscrive docs/grafo.md dal codice

# --- migration ---
uv run alembic upgrade head                     # applica
uv run alembic downgrade -1                     # torna indietro
uv run alembic current | history | heads        # stato
uv run alembic revision --autogenerate -m "..." # nuova migration dai modelli
```

---

## 5. Le API REST — elenco completo

Base URL: `http://localhost:8000`. Ogni risposta di errore ha la stessa **busta**:

```json
{
  "timestamp": "2026-10-15T10:00:00",
  "status": 404,
  "error": "NOT_FOUND",
  "message": "descrizione umana",
  "path": "/api/ai/glossary",
  "details": null
}
```

Ogni risposta porta gli header `X-Request-Id` (puoi inviarlo tu, se è `[A-Za-z0-9-]{8,64}`,
altrimenti viene generato un UUID) e `X-Process-Time`.

### 5.1 Sonda e documentazione

#### `GET /health`

Nessuna auth. Dice se il servizio è su, l'ambiente e ** quali** chiavi API sono presenti (mai il
valore).

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "UP",
  "timestamp": "2026-10-15T10:00:00",
  "app_name": "LipariBank AI",
  "version": "1.0.0",
  "environment": "development",
  "credentials": {"openai": true, "anthropic": false}
}
```

Anche `GET /docs` (Swagger), `GET /redoc` e `GET /openapi.json` sono pubblici.

---

### 5.2 Autenticazione

#### `POST /api/auth/login`

Nessuna auth. Corpo `application/x-www-form-urlencoded` (come OAuth2 password flow):
`username` e `password`.

```bash
curl -s -X POST http://localhost:8000/api/auth/login \
  -d "username=mbianchi&password=LA_TUA_SEED_PASSWORD"
```

Risposta 200:

```json
{"access_token": "eyJhbGciOi...", "token_type": "bearer"}
```

Il token è un JWT HS256 con claim `sub` (username), `role`, `iat`, `exp` e dura **30 minuti**.
Non c'è refresh: si rifà il login.

Errori: **401** `UNAUTHORIZED` «Credenziali non valide» (stessa risposta per password sbagliata e
utente inesistente — timing equalizzato), **422** se mancano i campi.

#### `POST /api/auth/token-mcp`

Serve un token REST valido (header `Authorization: Bearer …`) più il segreto del client.
Scambia il token dell'operatore con uno valido **solo** per un servizio pubblicato (MCP o A2A).

Corpo JSON:

| campo | tipo | note |
|---|---|---|
| `client_id` | str | deve essere registrato: `assistente-filiale`, `supporto-procedure`, `agente-compliance` |
| `client_secret` | str | HMAC-SHA256 non memorizzato: `uv run python -m src.auth.mcp_tokens <client_id>` lo stampa |
| `resource` | str | URI del servizio: `http://127.0.0.1:8200/mcp` (MCP) o `http://127.0.0.1:8300` (A2A) |

```bash
# 1. token operatore
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -d "username=mbianchi&password=..." | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 2. segreto del client
SECRET=$(uv run python -m src.auth.mcp_tokens assistente-filiale)

# 3. scambio
curl -s -X POST http://localhost:8000/api/auth/token-mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"client_id\":\"assistente-filiale\",\"client_secret\":\"$SECRET\",\"resource\":\"http://127.0.0.1:8200/mcp\"}"
```

Risposta 200:

```json
{"access_token": "eyJhbGciOi...", "token_type": "bearer", "expires_in": 600, "scope": "conti:leggi policy:leggi"}
```

Dura **10 minuti**. Gli scope sono l'**intersezione** di tre insiemi: quelli del client × quelli
del ruolo × quelli della risorsa. Un token MCP non vale sulla REST e viceversa.

Errori: **401** `CLIENT_NON_AUTENTICATO` (client sconosciuto o segreto sbagliato, stessa
risposta), **403** `TOKEN_NEGATO` (servizio non previsto per il client, o ruolo senza scope).

---

### 5.3 Chat

#### `POST /api/ai/chat`

Qualsiasi ruolo. La chat completa con tool-calling (può leggere i saldi) e storia di
conversazione.

Corpo JSON:

| campo | tipo | vincoli |
|---|---|---|
| `session_id` | str | `"new"` crea una sessione |
| `message` | str | 1–2000 caratteri |

```bash
curl -s -X POST http://localhost:8000/api/ai/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id":"new","message":"Ciao, come stai?"}'
```

Risposta 200:

```json
{
  "session_id": "uuid",
  "reply": "testo della risposta",
  "tool_calls": [{"name": "get_account_balance", "arguments": {"account_id": "IT60..."}, "result": "Saldo di ..."}],
  "tokens_used": 250,
  "cost_eur": 0.00004,
  "model_used": "gpt-4o-mini",
  "created_at": "2026-10-15T10:00:00"
}
```

Il giro: la richiesta passa al servizio chat → si carica la storia della sessione → il modello
può chiamare il tool del saldo → ogni chiamata è loggata in `tool_calls` → la risposta e i costi
finiscono nella sessione e nel ledger `llm_calls`. La sessione di un altro utente è **404**.

Prima di chiamare il modello c'è il **tetto di spesa giornaliero** (`daily_budget_eur`, default
5 €): se sforato → **429** `RATE_LIMIT` con `Retry-After` = secondi a mezzanotte UTC.

Altri errori: 401, 404 `CHAT_SESSION_NOT_FOUND`, 404 `ACCOUNT_NOT_FOUND` (se il tool viene
aggirato), 502 `LLM_PROVIDER_ERROR`, 503 `LLM_NOT_CONFIGURED` (manca la chiave).

#### `POST /api/ai/chat/stream`

Stesso corpo e regole, ma risposta in **SSE** (`text/event-stream`): righe
`data: {"text": "pezzo"}` e terminatore `data: [DONE]`. Gli errori (404/429) escono **prima**
che lo stream parta. Niente tool-calling in streaming.

---

### 5.4 Glossario

#### `POST /api/ai/glossary`

**Pubblico**, niente token. Spiega un termine interno da una tabella statica (`iban`, `sepa`,
`mav`).

```bash
curl -s -X POST http://localhost:8000/api/ai/glossary \
  -H "Content-Type: application/json" -d '{"term":"iban"}'
```

```json
{"term": "IBAN", "definition": "...", "source": "glossario interno"}
```

Termine sconosciuto → **404** `NOT_FOUND` «Il termine «X» non è nel glossario interno».

---

### 5.5 Categorizzazione

#### `POST /api/ai/categorize`

**Pubblico** (anonimo, il costo finisce nel ledger col segnaposto `-`). Categorizza un movimento
con output strutturato (instructor).

Corpo JSON:

| campo | tipo | vincoli |
|---|---|---|
| `description` | str | 1–200 |
| `amount` | number | > 0 |
| `currency` | str | default `EUR`, pattern `^[A-Z]{3}$` |

```bash
curl -s -X POST http://localhost:8000/api/ai/categorize \
  -H "Content-Type: application/json" \
  -d '{"description":"Bonifico Enel Energia bolletta","amount":87.40,"currency":"EUR"}'
```

```json
{
  "category": "UTILITIES",
  "subcategory": "ENERGY",
  "confidence": 0.8,
  "reasoning": "..."
}
```

`category` è chiusa: `UTILITIES | GROCERIES | TRANSPORT | RESTAURANTS | ENTERTAINMENT | OTHER`.
Se il modello non rispetta lo schema dopo 2 tentativi → **502** `LLM_PROVIDER_ERROR`. C'è anche
il tetto di spesa giornaliero (429).

---

### 5.6 Importazione movimenti

#### `POST /api/ai/movements/import`

**Pubblico**. Importa un CSV di movimenti, riga per riga: le valide entrano, le scartate
escono spiegate per numero.

- multipart, campo `file`;
- intestazione esatta: `date,description,amount,currency`;
- date in `AAAA-MM-GG`, importo > 0.

```bash
curl -s -X POST http://localhost:8000/api/ai/movements/import \
  -F "file=@movimenti.csv"
```

```json
{
  "imported_count": 4,
  "problems": [
    {"row": 3, "reason": "la data va scritta AAAA-MM-GG"},
    {"row": 8, "reason": "importo: deve essere maggiore di 0"}
  ]
}
```

File vuoto, non UTF-8 o header sbagliato → **422** `INVALID_FILE`. File mancante → 422
`VALIDATION_ERROR` con `details: ["file: Field required"]`. Le righe scartate non sono errori.

---

### 5.7 Conti

#### `GET /api/ai/accounts/{account_id}/movements`

Qualsiasi ruolo, ma il **muro del portafoglio**: il conto deve appartenere a un cliente
dell'operatore che chiede (identità presa dal token, mai da un parametro). Query richieste:
`date_from` e `date_to` in `YYYY-MM-DD`.

```bash
curl -s "http://localhost:8000/api/ai/accounts/IT60X0542811101000000123/movements?date_from=2026-01-01&date_to=2026-12-31" \
  -H "Authorization: Bearer $TOKEN"
```

```json
[{"booking_date": "2026-07-01", "description": "Bonifico verso Panama", "amount": "-12000.00", "owner": "Paolo Ferri"}]
```

Conto altrui o inesistente → **stesso** 404 `ACCOUNT_NOT_FOUND` «Il conto X non è nel tuo
portafoglio» (il nome dell'intestatario vero non compare mai). Periodo rovesciato → **422**
`INVALID_PERIOD`.

#### `GET /api/ai/accounts/{account_id}/spending`

Idem, ma aggrega la spesa per mese.

```json
{
  "account_id": "IT60...",
  "date_from": "2026-01-01",
  "date_to": "2026-12-31",
  "months": [{"month": "2026-07", "spent": "15200.00"}],
  "total": "20402.22"
}
```

I totali sono `Decimal`: la somma al centesimo torna (un float darebbe `20402.220000000016`).

---

### 5.8 Consulenza RAG

#### `POST /api/ai/advice`

Qualsiasi ruolo. Il cuore del RAG: riscrive la domanda, cerca nei documenti **filtrando per
ruolo**, genera una risposta con citazioni.

Corpo JSON: `question` (5–1000 caratteri).

```bash
curl -s -X POST http://localhost:8000/api/ai/advice \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"Quanto costa un bonifico urgente?"}'
```

```json
{
  "answer": "Un bonifico urgente costa ... [commissioni_bonifico]",
  "citations": [
    {"document_id": "commissioni_bonifico", "chunk_id": "c1", "excerpt": "...", "similarity": 0.78}
  ],
  "tokens_used": 800,
  "cost_eur": 0.0001,
  "rewritten_query": "commissione bonifico urgente importo"
}
```

Il giro (pipeline a fasi):
1. **Riscrittura** della domanda col modello (con cache e ripiego: se il modello cade, passa la
   domanda originale non cachata);
2. **Ricerca vettoriale** su `document_chunks` con soglia **0.35** e filtro
   `visibility = ANY(:livelli)` — i livelli del ruolo (vedi [§6.2](#62-visibilità-dei-documenti));
3. Se nessun passaggio supera la soglia → **rifiuto**, `citations: []` e **zero** chiamate al
   modello;
4. **Generazione** con citazioni `[document_id]` nel testo; se il modello cade o tempo-scade
   (10 s) → **200 parziale** «⚠️ Risposta parziale» con le fonti del solo recupero;
5. Solo le citazioni usate davvero nella risposta tornano indietro (un marcatore inventato o
   ripetuto non diventa fonte).

Limiti: **15 richieste al minuto per utente** (slowapi, keyed sul `sub` del token) — la 16ª è
**429** con `Retry-After: 60` — più il tetto di spesa giornaliero.

#### `POST /api/ai/documents/ingest`

Solo `compliance_lead` e `admin` (altrimenti 403). Scrive un documento nell'indice: chunking +
embedding + upsert per `document_id` (re-ingerire lo stesso id sostituisce i vecchi chunk).

Corpo JSON:

| campo | tipo |
|---|---|
| `document_id` | str, max 100 |
| `content` | str, min 10 |
| `metadata` | dict opzionale |
| `visibility` | `public` \| `internal` \| `risk_only` \| `compliance_only` (default `public`) |

```json
{"chunk_count": 3, "embedding_dim": 1536}
```

---

### 5.9 Agente

#### `POST /api/ai/agent`

Qualsiasi ruolo. Il ciclo dell'agente scritto a mano (`src/agents/loop.py`): max **6 passi**,
budget **0.05 €** per run, cinque tool (vedi [§7.1](#71-i-cinque-tool)).

Corpo JSON: `message` (1–2000).

```bash
curl -s -X POST http://localhost:8000/api/ai/agent \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message":"Qual è il saldo del conto IT60X0542811101000000123?"}'
```

```json
{
  "run_id": "uuid",
  "reply": "Il saldo del conto è 48200.00 EUR.",
  "steps": 2,
  "tool_calls": ["get_account_balance"],
  "stopped_by": "model",
  "cost_eur": 0.0003
}
```

Il giro del ciclo:
1. Il modello riceve i 5 tool e la domanda; ogni passo può chiamare più tool (le **letture**
   girano in parallelo, le **scritture** una alla volta);
2. Se un tool richiede approvazione e c'è un run: **nessun tool del passo parte** — lo stato va
   in `agent_runs` e `stopped_by="awaiting_approval"`, la risposta dice che nulla è stato
   eseguito;
3. Un tool che fallisce torna al modello come osservazione (mai il traceback);
4. Fine: `stopped_by` ∈ `model` (risposta pronta) | `max_steps` | `budget` | `awaiting_approval`
   — gli ultimi tre non sono errori HTTP, sono 200.

#### `GET /api/ai/agent/{run_id}`

Lo stato di un run. La vedono **chi ha chiesto** e i ruoli approvatori; per tutti gli altri è
404 (il run non è "non esistente", è "non tuo").

```json
{
  "run_id": "uuid",
  "status": "awaiting_approval",
  "requested_by": "mbianchi",
  "description": "Segnalazione sopra soglia: conto IT60..., importo 25000.00",
  "decided_by": null,
  "decisions": []
}
```

`status` ∈ `awaiting_approval | running | done | rejected`.

#### `POST /api/ai/agent/{run_id}/approve`

Solo `compliance_lead` e `risk_lead`, e **mai** chi ha chiesto (403 «Chi ha richiesto l'azione
non può deciderla»). Nessun corpo. Fa riprendere il run: esegue le chiamate sospese e restituisce
una `AgentResponse` completa.

```bash
curl -s -X POST "http://localhost:8000/api/ai/agent/$RUN_ID/approve" \
  -H "Authorization: Bearer $TOKEN_APPROVATORE"
```

Errori: 403 ruolo/autodecisione, 404 run inesistente o non visibile, **409** `CONFLICT`
«L'esecuzione non è più in attesa» (doppio clic: la seconda richiesta perde la gara).

#### `POST /api/ai/agent/{run_id}/reject`

Stesse regole di approve, ma il corpo chiede il motivo (`motivo`, 10–500 caratteri). Il run
riprende e l'agente riferisce il rifiuto al modello («non ritentarla»): zero scritture.

```json
{"motivo": "Importo non giustificato: serve documentazione prima"}
```

#### `POST /api/ai/supervisor`

Qualsiasi ruolo. La stessa domanda divisa: un triage sceglie la strada, due specialisti
lavorano (dati conto / policy), una sintesi compone.

```json
{
  "risposta": "...",
  "instradamento": "dati_conto",
  "specialisti_completi": {"dati_conto": true, "policy": true},
  "cost_eur": 0.001
}
```

`instradamento` ∈ `dati_conto | policy | entrambi`. Qui nessuno scrive segnalazioni: lo fa solo
il grafo nel nodo `scrivi`.

---

### 5.10 Admin

#### `GET /api/admin/cost-report`

Solo `risk_lead` e `admin`. Quanto costa il sistema, aggregato. Query: `dal` (data, obbligatoria).

```bash
curl -s "http://localhost:8000/api/admin/cost-report?dal=2026-10-01" \
  -H "Authorization: Bearer $TOKEN_RISK"
```

```json
{
  "dal": "2026-10-01",
  "totale_eur": "12.3456",
  "per_modello": [{"chiave": "gpt-4o-mini", "chiamate": 120, "token": 450000, "costo_eur": "12.34"}],
  "per_utente": [{"chiave": "mbianchi", "chiamate": 40, "token": 150000, "costo_eur": "4.00"}],
  "per_endpoint": [{"chiave": "advice", "chiamate": 30, "token": 100000, "costo_eur": "3.10"}],
  "run_piu_cari": [{"chiave": "uuid", "chiamate": 1, "token": 90000, "costo_eur": "0.90"}]
}
```

---

### 5.11 Matrice auth → endpoint

| Endpoint | Token | Ruolo extra |
|---|---|---|
| `GET /health`, `/docs`, `/redoc`, `/openapi.json` | no | — |
| `POST /api/auth/login` | no | — |
| `POST /api/ai/glossary`, `/api/ai/categorize`, `/api/ai/movements/import` | no | — |
| `POST /api/auth/token-mcp` | sì (REST) | + client_secret |
| `POST /api/ai/chat`, `/chat/stream`, `/advice`, `/agent`, `/supervisor` | sì | qualsiasi |
| `GET /api/ai/accounts/...`, `GET /api/ai/agent/{id}` | sì | qualsiasi (+ proprietà) |
| `POST /api/ai/agent/{id}/approve`, `/reject` | sì | `compliance_lead`, `risk_lead` (no autodecisione) |
| `POST /api/ai/documents/ingest` | sì | `compliance_lead`, `admin` |
| `GET /api/admin/cost-report` | sì | `risk_lead`, `admin` |
| `POST /a2a` (app A2A, porta 8300) | sì (audience A2A + scope `istruttoria:chiedi`) | — |
| `GET /.well-known/agent-card.json` (A2A) | no | — |

Un solo rate limit HTTP nell'app: **15/min su `/api/ai/advice`**. Poi il tetto di spesa
giornaliero (429) su `chat`, `chat/stream`, `advice`, `categorize`.

---

## 6. Autenticazione e ruoli

### 6.1 Flusso

```
login (form)  →  access token REST (30 min)  →  chiamate REST
                       │
                       └─ token-mcp (+client_secret)  →  token MCP/A2A (10 min, audience dedicata)
```

I ruoli esatti: `operator`, `compliance_lead`, `risk_lead`, `admin` (nessun seed), `public`
(imPLICITO se il token non ha il claim `role`). Il check è `require_role(*ammessi)` → 403 con
«Accesso riservato a: …».

### 6.2 Visibilità dei documenti

`src/auth/acl.py` mappa ruolo → livelli ammessi nel retrieval:

| Ruolo | Vede |
|---|---|
| `admin` | `public`, `internal`, `risk_only`, `compliance_only` |
| `compliance_lead` | `public`, `internal`, `compliance_only` |
| `risk_lead` | `public`, `internal`, `risk_only` |
| `operator` | `public`, `internal` |
| sconosciuto | `public` (con log `ruolo_sconosciuto`) |

Il filtro sta **nella query SQL**, prima dell'ordinamento: un canarino messo in un documento
`compliance_only` non arriva mai a un operatore, nemmeno nel prompt del modello.

### 6.3 Client MCP registrati

| client_id | risorse ammesse | scope del client |
|---|---|---|
| `assistente-filiale` | MCP + A2A | `conti:leggi`, `policy:leggi`, `istruttoria:chiedi` |
| `supporto-procedure` | solo MCP | `policy:leggi` |
| `agente-compliance` | solo A2A | `istruttoria:chiedi` |

Scope del ruolo: `operator` → tutti e tre; `compliance_lead` → `policy:leggi` +
`istruttoria:chiedi`; `risk_lead` → `policy:leggi`. Il segreto non è memorizzato:
`HMAC-SHA256(jwt_secret, "client:<id>")`.

---

## 7. L'agente: tool, approvazione, supervisor, grafo

### 7.1 I cinque tool

Tutti costruiti con `build_tools_for(user, deps)`: l'operatore è **nella closure**, non è un
argomento — non c'è modo di chiedere il dato di un altro.

| # | Tool | Cosa fa | Input | Output | Scrive? |
|---|---|---|---|---|---|
| 1 | `find_customer_accounts` | conti di un cliente del portafoglio | `customer_id` | «Conti del cliente X:…» o rifiuto | no |
| 2 | `get_account_balance` | saldo di un conto | `account_id` (IBAN) | «Saldo di …: 48200.00 EUR» o rifiuto | no |
| 3 | `list_recent_movements` | ultimi N movimenti | `account_id`, `n` (1–20, default 5) | righe «dd/mm descrizione +importo» | no |
| 4 | `search_documents` | RAG con ACL del ruolo, top 3 | `query` (min 3) | passaggi «[doc_id] testo…» o «NIENTE_NEI_DOCUMENTI» | no |
| 5 | `apri_segnalazione_compliance` | apre una pratica | `account_id`, `motivo` (10–500), `importo` | «Segnalazione <id> aperta…» | **sì** |

Regole chiave:

- **Muro del portafoglio**: cliente/conto altrui e cliente inesistente rispondono **uguali**
  («Non risulta nel portafoglio di questo operatore.»), così non si scopre chi esiste. Il
  rifiuto è loggato con l'**impronta** (hash) del valore, mai col valore.
- **Idempotenza**: due segnalazioni uguali fanno una riga (`idempotency_key` = conto+autore+data+
  importo, vincolo unique).
- **Solo la scrittura è vietata ai log**: `tool_args` per tool che scrivono è `<omessi>`, il log
  `agent_step` registra l'esito (`ok|vuoto|errore`) e la ricevuta, mai IBAN o saldi.

### 7.2 L'approvazione (human-in-the-loop)

Soglia: `settings.soglia_approvazione_eur` = **5000 €**. Sopra (o se l'importo non c'è) la
segnalazione si ferma:

1. Il tool che scrive **non parte** — nemmeno le letture dello stesso passo;
2. `stopped_by="awaiting_approval"`, il run sta in `agent_runs` con le pending_calls;
3. Un approvatore (`compliance_lead`/`risk_lead`, diverso dal richiedente) fa
   `approve`/`reject`;
4. `riprendi()` **ricostruisce i tool per il richiedente** (mai per chi approva) e riparte;
   un rifiuto manda al modello «non cercare un'altra strada» con il motivo;
5. La sosta **sopravvive al riavvio** del processo: lo stato è nel database.

### 7.3 Il grafo del supervisor (LangGraph)

`src/lg/`: nodi `triage`, `dati_conto`, `policy`, `prepara`, `chiedi`, `scrivi`, `istruttoria`,
`sintesi`, `chiusura`. La tabella **PERCORSI** dichiara il flusso:

| instradamento | nodi |
|---|---|
| `dati_conto` | `dati_conto` |
| `policy` | `policy` |
| `entrambi` | `dati_conto → policy` |
| `segnalazione` | `dati_conto → policy → prepara → chiedi → scrivi` |
| `istruttoria` | `dati_conto → istruttoria` |

- `instrada` mette **prima i tetti** (budget 0.05 €, max 12 passi → `chiusura`), poi segue il
  percorso;
- `chiedi` è sincrono: sotto soglia si auto-approva, sopra chiama `interrupt()` — il grafo si
  ferma e il **thread** (`f"{username}:{uuid4()}"`, generato server-side) aspetta nel
  checkpointer `AsyncPostgresSaver`;
- `scrivi` è l'unico nodo con effetto: chiama lo stesso tool `apri_segnalazione_compliance`;
- il modello si paga in `prepara`, **prima** della sosta: alla ripresa non si ripaga;
- le tabelle del checkpointer stanno **fuori dalle migration** (Alembic non deve toccarle).

La spiegazione passo passo è in `spiegazione_grafo.md`; il disegno generato dal codice è in
`docs/grafo.md` (rigenerabile con `uv run python -m scripts.disegna_grafo`).

### 7.4 Le eval delle quattro implementazioni

`uv run python -m evals.confronto` misura `ciclo`, `catena`, `agente`, `grafo` sugli stessi 9
casi. Ultimo rilievo (README): ciclo 0.89, agente 0.89, catena 0.22, grafo 0.33 — il grafo
paga l'ordine (triage + sintesi sono chiamate vere) ma passa i casi multi-hop.

---

## 8. Il server MCP

Due server:

| Server | Trasporto | Comando | Porta |
|---|---|---|---|
| stdio | `LIPARI_TOKEN` nell'ambiente | `uv run python -m liparibank_mcp.server` | — |
| pubblico | streamable-http, token JWT | `uv run python -m liparibank_mcp.pubblico` | **8200**, endpoint `/mcp` |

### 8.1 Scoperta senza token

```bash
curl -si http://127.0.0.1:8200/mcp
# 401 + WWW-Authenticate: Bearer resource_metadata="http://127.0.0.1:8200/.well-known/oauth-protected-resource/mcp"

curl -s http://127.0.0.1:8200/.well-known/oauth-protected-resource/mcp
# {"resource":".../mcp","authorization_servers":["http://127.0.0.1:8000/"],
#  "scopes_supported":["conti:leggi","policy:leggi"], ...}
```

Non c'è registrazione OAuth dinamica: il client si registra solo nella tabella `CLIENTI`, scambia
il token via `POST /api/auth/token-mcp` sulla API a 8000, e quel token (audience `…/mcp`) è
l'unico che entra. Un token senza `exp`, di audience sbagliata o con altra firma è 401
`invalid_token`.

### 8.2 Tool pubblicati (server pubblico)

| Tool | Versioni | Scope | Output |
|---|---|---|---|
| `search_policy` | v1 (`query`), v2 (`query`, `quanti` 1–5) | `policy:leggi` | testo dei passaggi; v2 controlla quanti |
| `find_customer_accounts` | **v1 testo**, **v2 JSON** | `conti:leggi` | v1: stringa di sempre; v2: `{"cliente":…,"conti":[{"iban":…,"etichetta":…}]}` |
| `get_account_balance` | non versionato | `conti:leggi` | stringa del saldo |

Regole del versioning (la "strada C"): chi **non sceglie** la versione prende la **più alta**
(quindi la v2 JSON — un client che parsava il testo si rompe, ed è voluto); la v1 resta servita
con data di disdetta dichiarata in `registry/server.json`
(`disdette.find_customer_accounts.1 = "2026-11-15"`). Il test `test_la_disdetta_registrata_corrisponde_al_server_vivo`
verifica che il registry non menta rispetto al server vivo.

Muro del portafoglio e filtro dei documenti valgono **anche da fuori**: un token MCP non li
aggira.

### 8.3 Superficie

`uv run python -m evals.superficie` fa girare il server vero in processo e, per ogni coppia
client×ruolo, emette il token e stampa i tool visibili: 9 coppie, 8 tool esposti, 0 che
scrivono. È la tabella che legge un auditor.

---

## 9. L'agente A2A

`uv run python -m liparibank_a2a.agente` → `http://127.0.0.1:8300`.

- **Carta pubblica** `GET /.well-known/agent-card.json`: nome `istruttoria-lipari`, skill
  `istruttoria` (tag compliance/aml/movimenti), sicurezza Bearer JWT;
- **RPC** `POST /a2a` (JSON-RPC 2.0): protetto da middleware — serve token con
  `aud == http://127.0.0.1:8300`, issuer giusto, `exp` presente e scope **`istruttoria:chiedi`**;
- Comportamento: se nel testo manca un IBAN, il task resta aperto in `INPUT_REQUIRED` con
  «Su quale conto? Serve l'IBAN.»; altrimenti gira il **grafo della banca** per l'operatore del
  token e chiude con artefatto `istruttoria`; un errore produce `TASK_FAILED` con messaggio
  generico;
- Un task di un operatore non è visibile a un altro.

Esempio di chiamata (dopo aver ottenuto un token A2A con `resource = http://127.0.0.1:8300`):

```bash
curl -s -X POST http://127.0.0.1:8300/a2a \
  -H "Authorization: Bearer $TOKEN_A2A" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"role":"user","parts":[{"type":"text","text":"Istruttoria sul conto IT60X0542811101000000123"}]}}}'
```

Il lato client della delega (il grafo che chiede all'agente A2A) è in `src/lg/delega.py`: max 3
giri (`TROPPI_GIRI`), task perso → esito `TASK_PERSO`.

---

## 10. I test — infrastruttura ed elenco completo

**203 test** (≈ 220 con le parametrize). Non chiamano **nessun modello**: usano i finti di
`tests/finti.py`. Comando base:

```bash
uv run pytest -q
```

### 10.1 Come è costruita la suite

**`conftest.py` (radice)** — ambienta ogni run:

- forza `DATABASE_URL` sul DB di test (`…/lipari_ai_test`, o `TEST_DATABASE_URL` se la imposti)
  e le chiavi API finte (`sk-test-mai-valida`) — così un test dimentico non tocca il DB di
  sviluppo né spende denaro;
- fixture di sessione `schema()`: `alembic downgrade base` + `upgrade head` → **ogni pytest
  richiede un Postgres raggiungibile**;
- override dell'app: LLM → `ModelloEco` («Echo: …», gratis), categorizzazione → a regole,
  embedder → finto, riscrittura → spenta, utente → sempre `mbianchi` operatore.

**`tests/conftest.py`** — le fixture vere:

- `session`: DB **sqlite in memoria** con le tabelle "del giorno" (`customers`, `accounts`,
  `movements`, `compliance_alerts`, `agent_runs`, `llm_calls`), 2 clienti e 2 conti seminati —
  è il banco dei test leggeri;
- `deps_reali`: i repository veri sulla sqlite + `AsyncMock` come modello;
- `deps_spia`: `MagicMock` per assert «il repo è interrogato così»;
- costanti: `CLIENTE_DI_MARCO = "C-10234"` (nel portafoglio), `CLIENTE_ALTRUI = "C-20417"`
  (di `pgalli`), `CONTO_DI_MARCO`/`CONTO_ALTRUI`.

**`tests/finti.py`** — la cassa di risonanza dei test:

| Finto | Cosa fa |
|---|---|
| `openai_finto` / `anthropic_finto` | SDK veri + `httpx.MockTransport`: il traffico HTTP è finto |
| `ModelloEco` | risponde «Echo: …» a costo zero |
| `ModelloFisso` | risponde un testo scelto dal test e **registra i prompt** ricevuti |
| `Copione` (nei test LangChain) | risponde a turno e registra ogni chiamata |
| `embedder_finto` | vettori 1536 dim. **coerenti** (hash della parola): le parole comuni avvicinano |
| `riscrittore_spento` | la domanda passa invariata |
| `CategorizzazioneARegole` | categorizzazione deterministica a parole chiave |

**`tests/banco_app.py` + `tests/riavvio_altro_processo.py`** — il banco dei test "sopravvive al
riavvio": una FastAPI sullo stesso router dell'agente con DB sqlite **su file** e utente letto
dall'header `X-Utente` (header di solo-test, non esiste in produzione); lo script figlio riapre
il DB e chiama `approve` da un **secondo processo Python**.

**Esecuzione**: `asyncio_mode = "auto"` (ogni `def async` è un test), un solo event loop di
sessione. **Nessun marker custom**: i sottoinsiemi si scelgono con percorsi e `-k`.

### 10.2 Postgres vs sqlite

- **Postgres sempre**, per il fixture `schema()` — anche quando il test poi lavora su sqlite;
- **direttamente su Postgres**: `test_g1_strada_a`, `test_g1_langchain` (fixture `banca`),
  `test_g2_langgraph` (checkpointer), `test_g3`, `test_g3_dura`, `test_g4`, `test_g4_strada_c`,
  `test_g4_pubblicare`, `test_g5`, `test_g6`, `test_g9`, `unit/test_mcp` (un caso);
- **sqlite in memoria**: quasi tutto `unit/test_g7`, tre casi di `unit/test_g8`;
- **sqlite su file**: 10 casi di `unit/test_g8` (sosta + riavvio), il sottoprocesso di
  `unit/test_mcp`.

### 10.3 Elenco per file

#### `tests/test_g1.py` — configurazione e sonda (4)

| Test | Cosa insegna |
|---|---|
| `test_senza_variabili_obbligatorie_il_processo_si_ferma_e_dice_quali` | config mancante → exit 1 con «manca», niente traceback |
| `test_un_valore_del_tipo_sbagliato_nomina_il_campo` | il ValidationError nomina il campo |
| `test_un_segreto_corto_non_basta` | `JWT_SECRET` corto rifiutato |
| `test_la_sonda_dice_quali_chiavi_ci_sono_senza_mostrarle` | `/health` non mostra mai la chiave |

```bash
uv run pytest tests/test_g1.py -q
```

#### `tests/test_g1_langchain.py` — le quattro implementazioni sul serio (11)

| Test | Cosa insegna |
|---|---|
| `test_tre_implementazioni_sugli_stessi_casi` | ciclo e agente 1.0, catena 1/3, costi e chiusura dei run |
| `test_la_sosta_dell_agente_decide_sugli_argomenti_validati` | la sosta vede gli argomenti grezzi |
| `test_la_catena_non_cita_una_fonte_che_il_retrieval_non_ha_dato` | niente fonti inventate |
| `test_il_tetto_dell_agente_non_diventa_una_risposta` | max_steps ≠ risposta |
| `test_dopo_la_sosta_decide_una_persona` (4 param) | approve → 1 alert, reject → 0 |
| `test_un_uscita_fuori_schema_e_un_errore_e_il_costo_resta` | errore → costo comunque pagato |
| `test_la_catena_non_porta_al_modello_un_documento_che_il_ruolo_non_vede` | ACL anche nel prompt |
| `test_l_agente_non_vede_un_cliente_di_un_altro_portafoglio` | rifiuto senza dati |
| `test_un_tool_che_si_rompe_torna_al_modello_come_osservazione` | errore tool → ToolMessage |
| `test_due_letture_nello_stesso_passo_vanno_in_fila` | coda serializza sulla sessione |
| `test_il_tetto_dell_agente_conta_anche_i_passi_prima_della_sosta` | i passi delle fasi contano |

```bash
uv run pytest tests/test_g1_langchain.py -q
```

#### `tests/test_g1_strada_a.py` — tetto ricerche, controlli eval (5)

`test_tetto_ricerche`, `test_il_caso_multi_hop_legittimo_sta_sotto_il_tetto`,
`test_sesto_controllo_valore` (`risposta_contiene`), `test_casi_nel_dataset`,
`test_nome_entro_300`.

#### `tests/test_g2.py` — contratto HTTP (16)

Il banco degli errori: busta unificata, 405, 500 senza dettagli interni, 429 con `Retry-After`,
CORS, request-id, categorizzazione a regole, CSV (riga per riga, BOM Excel, header sbagliato →
mai un 500).

```bash
uv run pytest tests/test_g2.py -q
```

#### `tests/test_g2_langgraph.py` — il grafo (12)

Il più importante per il supervisor:

| Test | Cosa insegna |
|---|---|
| `test_davanti_al_banco_il_grafo_paga_l_ordine_sulle_domande_semplici` | il grafo 1/3 sui casi semplici: il triage costa passi |
| `test_il_grafo_legge_prima_i_conti_poi_il_saldo` | `visitati == [triage, dati_conto, sintesi]` |
| `test_sotto_soglia_il_grafo_scrive_senza_chiedere` | nessun interrupt, 1 alert |
| `test_sopra_soglia_il_grafo_si_ferma_e_una_persona_scrive` | interrupt + resume approvato → 1 alert |
| `test_una_respinta_non_scrive_e_lo_dice_nella_risposta` | rifiuto nella risposta |
| `test_la_sosta_sopravvive_a_un_checkpointer_nuovo` | checkpointer nuovo stesso DB → riparte |
| `test_la_tabella_dichiara_un_insime_chiuso_di_percorsi` | PERCORSI = 5 percorsi dichiarati |
| `test_i_tetti_vengono_prima_di_qualunque_scelta` | budget o passi → `chiusura` |
| `test_la_chiusura_componi_quello_che_c_e_non_puo_fallire` | `chiudi` senza modello |
| `test_sopra_il_budget_il_grafo_chiude_senza_modello` | 1 chiamata e stop |
| `test_il_thread_porta_l_username_e_un_uuid` | `thread_id` = `username:uuid` |
| `test_le_tabelle_del_checkpointer_non_nelle_migration` | escluse dall'autogenerate |

```bash
uv run pytest tests/test_g2_langgraph.py -q
uv run pytest tests/test_g2_langgraph.py -q -k sosta
uv run pytest tests/test_g2_langgraph.py -q -k tetti
```

#### `tests/test_g3.py` — persistenza (10)

Conversazione che sopravvive al riavvio, 404 sulla sessione altrui, transazione atomica, 20
messaggi letti in 2 query, migration reapplicabile, seed idempotente, movimenti con
intestatario, totali in Decimal, muro del portafoglio, periodo rovesciato 422.

#### `tests/test_g3_dura.py` — memoria durevole (20)

Fatti estratti dal testo che restano per 50 turni, contesto sotto tetto, sosta dentro la
conversazione, **preferenze dell'operatore** in uno store con namespace e TTL (Postgres
compreso), budget del ragionamento (`reasoning_effort`), stream di avanzamento senza dati
sensibili, ramo di prova `prova_da` che non scrive.

#### `tests/test_g4.py` — provider e chat (13)

Traduzione delle chiamate OpenAI **e** Anthropic (stesso giro tool, due lingue), stream con
`include_usage`, retry sui soli errori transitori, factory (modello sconosciuto → ValueError),
storia della conversazione, tetto a zero → 429 **prima** di chiamare il modello, instructor con
2 tentativi.

#### `tests/test_g4_strada_c.py` — il versioning (6)

```bash
uv run pytest tests/test_g4_strada_c.py -q
```

| Test | Cosa insegna |
|---|---|
| `test_le_due_versioni_di_find_customer_accounts_convivono` | v1 testo e v2 JSON sotto lo stesso nome |
| `test_senza_versione_arriva_la_piu_alta_e_questo_rompe_il_vecchio` | il default è la v2: è la rottura voluta |
| `test_il_muro_del_portafoglio_vale_anche_sulla_v2` | `conti: []` per il cliente altrui |
| `test_la_voce_di_registry_dice_le_versioni_e_la_disdetta` | registry coerente |
| `test_la_disdetta_registrata_corrisponde_al_server_vivo` | il registry non mente |
| `test_superficie_c_vede_le_due_versioni_sotto_lo_stesso_tool` | `find_customer_accounts v2/1` |

#### `tests/test_g4_pubblicare.py` — token, MCP, A2A, delega (31)

Il più esterno: scope = intersezione, token brevi e bound al client, scambio rifiutato (5 casi),
token da un servizio non valido sull'altro, server MCP che accetta solo il **suo** token,
ogni client vede i tool del suo scope, muro del portafoglio e dei documenti **anche da fuori**,
eccezioni che non raccontano, registry valido contro lo schema, superficie, agent card, task che
chiede l'IBAN, isolamento dei task per operatore, delega del grafo con tetto 3 giri, seed
password (rifiuta vuota/`bootcamp`/corta, `--ruota` obbligatorio per cambiarla, mai stampata).

#### `tests/test_g5.py` — RAG (10)

Chunking con overlap senza perdere niente, re-ingest = sostituzione, rifiuto fuori corpus senza
chiamare il modello, citazioni solo su ciò che la risposta usa, marcatore inventato scartato,
soglia 0.35, endpoint ingest+advice.

#### `tests/test_g6.py` — auth, ACL, riscrittura, limiti (14)

Login che non dice quale credenziale è sbagliata (+ dummy hash per il timing), token alterato /
scaduto / `alg none` respinti, ruolo sconosciuto → solo `public` nel log, canarino `ortensia-viola-sette`
visibile solo alla compliance, filtro SQL prima dell'ORDER BY, ripiego del riscrittore non
cachato, risposta parziale con `used_fallback`, limite 15/min per utente, chat e conti
identificati dal token.

#### `tests/unit/test_g7.py` — tool e ciclo (22)

Muro del portafoglio (cliente altrui = inesistente), log con impronta e `<omessi>`, idempotenza
delle segnalazioni, tetto passi e budget, tool non offerto → «Forse intendevi '…'», modello
sconosciuto → 0 chiamate, endpoint `/api/ai/agent`, log senza mai IBAN/saldi.

```bash
uv run pytest tests/unit/test_g7.py -q
uv run pytest tests/unit/test_g7.py::test_saldo_di_un_conto_del_portafoglio -q
```

#### `tests/unit/test_g8.py` — human-in-the-loop (13)

Sopra soglia si ferma e non scrive; sotto scrive; nello stesso passo **nulla** parte; doppia
sosta; chi chiede non approva; operatore non approva; doppio clic → `[200, 409]`; respinta non
ritenta; **approvazione sopravvive a un secondo processo**; triage e sintesi del supervisor.

#### `tests/unit/test_mcp.py` — server MCP (5)

Il server si descrive (3 tool `readOnlyHint`), identità non verificata rifiutata, server stdio
reale in sottoprocesso che rispetta il portafoglio, il ciclo agente usa i tool remoti senza
cambiare, il client passa `LIPARI_TOKEN` nell'ambiente.

#### `tests/unit/test_parallelo.py` — tool in parallelo (1)

Due letture partono insieme, lo strumento che scrive aspetta: «chi legge insieme chi scrive
dopo».

#### `tests/test_g9.py` — eval, costi, log (10)

Cancello (tutte le metriche sopra soglia, mancante → KeyError), exit code 0/1 per la CI, righe
in `llm_calls` per ogni chiamata che spende (modello gratis → zero righe), report costi riservato,
log JSON con `request_id` e traceback, request-id del chiamante che finisce in tutti i log.

### 10.4 Ricette veloci

```bash
# il test della sosta su Postgres, il più "completo"
uv run pytest tests/test_g2_langgraph.py::test_la_sosta_sopravvive_a_un_checkpointer_nuovo -q

# tutto il human-in-the-loop
uv run pytest tests/unit/test_g8.py -q

# il versioning MCP
uv run pytest tests/test_g4_strada_c.py -q

# solo i test che toccano un nome
uv run pytest -q -k portafoglio

# copertura del solo src
uv run pytest --cov=src --cov-report=term-missing -q
```

---

## 11. Le eval

Le eval **chiamano il modello vero**: servono chiavi valide e costa (frazioni di centesimo per
caso). Prima bisogna ingerire i documenti (`scripts.ingest_docs` oppure
`evals.ingest_fixtures`, che non passa dall'API).

| Comando | Cosa fa |
|---|---|
| `uv run python -m evals.runner` | 3 metriche su 3 dataset + **gate**: exit 1 se una metrica è sotto soglia |
| `uv run python -m evals.confronto` | le 4 implementazioni sugli stessi casi, con costo e ricerche per caso |
| `uv run python -m evals.superficie` | tabella client×ruolo × tool visibili, letta dal server vivo |
| `uv run python -m evals.durata` | conversazione di 50 turni: fatti, contesto, costo per blocco |
| `uv run python -m evals.ingest_fixtures` | ingestione diretta di `data/docs/` (per la CI, niente API) |

Metriche e soglie (`evals/runner.py`):

| Metrica | Dataset | Soglia | Controlli |
|---|---|---|---|
| `categorize` | `datasets/categorize.jsonl` | 0.80 | categoria esatta |
| `advice` | `datasets/advice.jsonl` | 0.70 | Recall@5 sul documento atteso; `doc_id: null` = deve rifiutare |
| `traiettorie` | `datasets/agent_traiettorie.jsonl` (9 casi) | 0.80 | `tool_richiesti`, `tool_vietati`, `passi ≤ passi_max`, `terminazione`, `citazione`; il confronto aggiunge `risposta_contiene` |

I run sospesi che le eval lasciano in bianco vengono chiusi con
`runs.chiudi(run_id, "closed_by_eval")`: niente pratiche vere dai test.

Output atteso (forma):

```text
traiettorie  0.89 (8/9)  soglia 0.80  ok            costo EUR 0.0027
    x traj-003: {"violati": ["citazione"], ...}
  traj-001   0 ricerche  0.000309 EUR
  ...
GATE: passato
```

---

## 12. Il database

Migration in `alembic/versions/` (catena: `82d54279bcb4` → … → `2da83670510f` head).

| Tabella | Scopo |
|---|---|
| `chat_sessions`, `chat_messages` | storico della chat per utente, con token e costo |
| `customers` | clienti; `operator` = chi li ha in portafoglio (indice) |
| `accounts` | conti (id = IBAN), saldo `numeric(12,2)` |
| `movements` | movimenti dei conti |
| `document_chunks` | chunk RAG: `embedding vector(1536)` con indice **HNSW** cosine, `visibility` |
| `app_users` | utenti con hash argon2 e ruolo |
| `compliance_alerts` | segnalazioni; `idempotency_key` UNIQUE |
| `agent_runs` | run sospesi/approvati: status, pending_calls, decisions (audit) |
| `llm_calls` | ledger dei costi: endpoint, utente, modello, token, costo, run |
| `checkpoints*`, `store*` | create da LangGraph al volo, **fuori dalle migration** |

---

## 13. Catalogo degli errori

| HTTP | `error` | Quando |
|---|---|---|
| 401 | `UNAUTHORIZED` | token mancante/scaduto/alterato, credenziali sbagliate |
| 401 | `CLIENT_NON_AUTENTICATO` | token-mcp: client sconosciuto o segreto sbagliato |
| 403 | `FORBIDDEN` | ruolo non ammesso, autodecisione in approve/reject |
| 403 | `TOKEN_NEGATO` | token-mcp: servizio non previsto o nessuno scope |
| 404 | `NOT_FOUND` / `CHAT_SESSION_NOT_FOUND` / `ACCOUNT_NOT_FOUND` | glossario, sessione altrui, conto fuori portafoglio |
| 405 | `METHOD_NOT_ALLOWED` | metodo sbagliato |
| 409 | `CONFLICT` | doppio approve/reject concorrente |
| 422 | `VALIDATION_ERROR` | Pydantic: `details` elenca i campi |
| 422 | `INVALID_FILE` / `INVALID_PERIOD` | CSV malformato / date invertite |
| 429 | `RATE_LIMIT` | 15/min su advice, oppure budget giornaliero esaurito |
| 500 | `INTERNAL_ERROR` | catch-all, senza dettagli interni |
| 502 | `LLM_PROVIDER_ERROR` | fornitore LLM/embedding, schema non rispettato |
| 503 | `LLM_NOT_CONFIGURED` | manca la chiave API per il modello richiesto |

---

## Per approfondire

- `spiegazione_grafo.md` — il grafo del supervisor, passo passo;
- `docs/grafo.md` — Mermaid + tabella PERCORSI, generati da `scripts.disegna_grafo`;
- `README.md` — misure, eval storiche, nota di revisione del codice scritto con l'AI.
