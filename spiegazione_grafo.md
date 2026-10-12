# Il grafo del supervisor: com'è fatto e come si prova

Il supervisor di LipariBank — la parte che decide chi lavora su una domanda, con quali tool e in
che ordine — è scritto come **grafo LangGraph** in `src/lg/`. Ogni passo del flusso è un nodo
nominato, il percorso è deciso da una tabella, e la sosta per l'approvazione di una persona vive
nel database: sopravvive a un riavvio del processo.

Questo documento dice **cosa** ho scritto, **dove** sta e **perché** ha quella forma. In fondo,
come si prova tutto — i comandi, i test, e cosa misurano.

## 0. Il problema, in una riga

Stesso contratto di prima (una domanda, i tool dell'operatore, i tetti, la sosta sopra soglia),
ma in una forma in cui il flusso è **dichiarato** — una tabella, non un'if annidata — e la sosta
è un `interrupt()` di LangGraph appoggiato a un checkpointer di Postgres.

Il motore HTTP (`src/api/agent.py`) continua a girare sul ciclo scritto a mano
(`src/agents/loop.py`): il grafo è la **quarta implementazione** dello stesso comportamento,
misurata dalle eval insieme alle altre tre e coperta da un banco di test dedicato
(`tests/test_g2_langgraph.py`).

## 1. Prima di tutto: lo stato (`src/lg/stato.py`)

Un grafo è una macchina a stati: lo stato è il primo pezzo da decidere. È un `TypedDict` —
non una classe con metodi — perché nel checkpointer finisce come JSON e perché nessun nodo
deve poterci «attaccare» logica.

I campi sono in due gruppi, e la differenza conta:

```python
# --- si accumulano: hanno il reducer add, nessuno può sovrascrivere il conto di un altro
contributi: Annotated[list[Contributo], add]
visitati: Annotated[list[str], add]
passi: Annotated[int, add]
costo_eur: Annotated[Decimal, add]

# --- l'ultimo fotogramma: nessun reducer, l'ultima scrittura vince
domanda, richiedente, instradamento, proposta, decisione, risposta, stopped_by
```

- `add` vuol dire che se due nodi scrivono nello stesso super-step i valori **sommano**: il
  passo che un nodo ha appena registrato non può essere annullato da chi viene dopo. `visitati`
  è una lista perché così si fonde senza perdere l'ordine in cui i nodi hanno lavorato.
- Il costo è in **euro `Decimal`**, non in micro-interi: qui il checkpointer salva solo la
  sosta, e il calcolo usa gli stessi `PREZZI_PER_1K` del ciclo a mano (`src/agents/loop.py`),
  letti dagli `usage_metadata` che LangChain mette su ogni risposta.
- `Contributo` è il contratto fra chi lavora e chi sintetizza: una riga JSON con `specialista`
  (un `Literal`), `completo`, `contenuto`, `tool_calls`. La sintesi riceve quei dati, non la
  prosa di un altro nodo.
- `Decisione` è la risposta della persona: `approvato`, `da`, `motivo`. Il nodo `chiedi` la
  compone da un `True` vero e basta — «solo un `True` è un sì» — così una risposta illeggibile
  diventa un no motivato, mai un sì per sbaglio.

## 2. Il flusso dichiarato: `PERCORSI` (`src/lg/grafo.py`)

Il cuore è una tabella di quattro righe:

| instradamento | nodi |
|---|---|
| `dati_conto` | dati_conto |
| `policy` | policy |
| `entrambi` | dati_conto → policy |
| `segnalazione` | dati_conto → policy → prepara → chiedi → scrivi |

L'ordinamento non è un dettaglio: i dati si raccolgono **prima** che si prepari la proposta, e
la proposta passa dalla sosta (`chiedi`) **prima** di arrivare alla scrittura (`scrivi`). Chi
cambia un flusso cambia una riga, e il disegno cambia con essa.

Due funzioni leggono la tabella:

- `instrada(state)` dice chi lavora dopo. **I tetti vengono prima di tutto**: se il costo ha
  superato `BUDGET_PER_RUN` o i passi hanno superato `MAX_PASSI` (12), la risposta è
  `chiusura`, senza nemmeno guardare il percorso. Poi scorre i nodi del percorso saltando
  quelli già `visitati`; `chiedi` e `scrivi` hanno inoltre bisogno di una `proposta` — se la
  preparazione non l'ha prodotta, si saltano e si va alla sintesi con quello che c'è.
- `successori(nodo)` calcola l'insieme **chiuso** degli archi in uscita (il nodo che segue nel
  percorso, più `sintesi` e `chiusura`): serve al `compile` e al disegno, e garantisce che non
  esista un arco a sorpresa.

La `chiusura` (`chiudi`) non chiama nessuno e non può fallire: compone quello che c'è — i
contributi completati e cosa resta da fare — e dichiara il motivo. Se sono sforati **tutti e
due** i tetti, vince il costo: è la regola che il test mette alla prova.

## 3. I nodi, uno per responsabilità

| nodo | cos'è |
|---|---|
| `triage` | una chiamata breve (`max_tokens=5`): una parola sola fra `dati_conto`, `policy`, `segnalazione`, `entrambi`. Una risposta fuori elenco prende `entrambi`, la strada prudente |
| `dati_conto`, `policy` | uno **specialista per volta**: `create_agent` con solo i SUOI tool (già in fila, uno alla volta) e `ModelCallLimitMiddleware(run_limit=4)` — finiti i passi, l'agente esce e il grafo va avanti |
| `prepara` | la chiamata strutturata (`SegnalazioneArgs`, `function_calling`): è QUI che si paga il modello, **prima** della sosta, così alla ripresa non si ripaga. Se gli argomenti non validano, lascia un contributo incompleto e il grafo salta la sosta |
| `chiedi` | il nodo **sincrono** della sosta: sotto soglia si approva da solo; sopra soglia chiama `interrupt()` con tutto e solo quello che serve a chi decide (`azione`, `conto`, `importo`, `motivo`, `richiesto_da`) |
| `scrivi` | l'unico nodo con un effetto: lo **stesso tool** `apri_segnalazione_compliance` del ciclo, con lo stesso muro e la stessa idempotenza. Se la decisione è no, il rifiuto (`RIFIUTO` di `src/agents/approval.py`) finisce nei contributi e nella risposta |
| `sintesi` | compone la risposta finale dai contributi in JSON, con un tetto di 500 token come la sintesi del supervisor |
| `chiusura` | descritta sopra: il tetto non è una risposta, e lo si capisce leggendolo |

Il grafo si costruisce con `costruisci_grafo(user, deps, llm, llm_triage, checkpointer)`: i tool
nascono **per chi ha chiesto** (`build_tools_for`) e vivono in closure, così il grafo non sa
nulla del database e in un test si compila con un `Deps` finto.

## 4. Dove il grafo si ferma: il checkpointer

Un grafo compilato senza checkpointer «funziona», ma la sosta non ha dove aspettare. Quindi:

- `checkpointer_postgres()` (`src/lg/grafo.py`) apre un `AsyncPostgresSaver` — il saver
  ufficiale di `langgraph-checkpoint-postgres` — per tutta la vita del servizio, e chiama
  `setup()`: le sue tabelle si creano una volta sola, idempotente.
- Il DSN dell'app parla `asyncpg`; qui serve `psycopg`: stessa stringa, driver diverso, e la
  sostituzione è una riga (`replace("postgresql+asyncpg://", "postgresql://")`).
- **Su Windows psycopg async vuole un SelectorEventLoop**: il ProactorEventLoop di default non
  ha `add_reader`. Nel servizio non è un problema; nel test del checkpointer sì, e
  `test_la_sosta_sopravvive_a_un_checkpointer_nuovo` accende un selettore tutto suo.
- Le quattro tabelle del checkpointer (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
  `checkpoint_migrations`) **non stanno nei modelli**, quindi Alembic non le conosce: senza
  un filtro, `revision --autogenerate` le troverebbe nel database e scriverebbe una migration
  che le cancella — con le soste che contengono. Il filtro è
  `fuori_dalle_migration` (sempre in `src/lg/grafo.py`), agganciato come `include_object` in
  **entrambe** le `context.configure` di `alembic/env.py`.

**Il thread è la richiesta**: `thread_per(user)` compone `f"{username}:{uuid4()}"`, generato
server-side. Due run non possono incrociarsi perché non condividono un identificatore, e
quello vecchio non si riutilizza. Alla ripresa il grafo riparte dal nodo `chiedi` — rilegge lo
stato, non richiama il modello — poi `scrivi` e `sintesi`.

## 5. Misurare: la quarta implementazione (`evals/`)

`evals/implementazioni.py` ha la classe `Grafo`, che gira sugli stessi casi delle altre tre; la
quarta riga di `evals/confronto.py::main()` la monta col modello del supervisor
(`MODELLO_TRIAGE`, cioè `settings.triage_model` dal `.env`). Numeri del **12/10/2026**
(`uv run python -m evals.confronto`, 9 casi, modello `openai/gpt-oss-120b` su Groq):

| implementazione | punteggio | costo EUR | ricerche |
|---|---|---|---|
| ciclo (a mano) | 0.89 (8/9) | 0.0024 | 6 |
| catena | 0.22 (2/9) | 0.0012 | 9 |
| agente prebuilt | 0.89 (8/9) | 0.0026 | 7 |
| **grafo** | **0.33 (3/9)** | **0.0045** | **6** |

Il grafo paga due chiamate che il ciclo non ha — il triage e la sintesi — e i `passi_max` del
dataset (1–6) sono tarati sul ciclo: il sovrapprezzo strutturale basta a farlo uscire su
`passi` su quasi tutti i casi a un salto, mentre passa i tre casi multi-hop col budget più
largo. Non è un difetto del flusso, è il prezzo della forma: il grafo compra flusso dichiarato,
specialisti con soli i loro tool, e una sosta nel database. Il banco di test fissa questo
comportamento con numeri finti e deterministici (vedi 7.2).

## 6. Il disegno (`scripts/disegna_grafo.py` → `docs/grafo.md`)

Il disegno non è fatto a mano: lo script compila il grafo (con servizi e LLM finti: nessun nodo
gira), ne estrae il Mermaid e lo scrive in `docs/grafo.md` con la tabella `PERCORSI`. Se
cambi una route o un arco, il disegno cambia con essa:

```bash
uv run python -m scripts.disegna_grafo
```

## 7. Come si prova tutto

### 7.1 I controlli di routine

```bash
uv run pytest -q                                   # 152 test, sul database lipari_ai_test
uv run mypy .                                      # strict, tutto il progetto
uv run ruff check . && uv run ruff format --check .
```

I test non chiamano nessun modello: le risposte finte hanno la forma vera dell'SDK
(`tests/conftest.py::risposta`), e il banco gira su Postgres (`lipari_ai_test`, dentro il
container `lipari-pg`), seminato a ogni test dal fixture `banca` di
`tests/test_g1_langchain.py`.

### 7.2 Solo il grafo: i dodici test (`tests/test_g2_langgraph.py`)

```bash
uv run pytest tests/test_g2_langgraph.py -q
uv run pytest tests/test_g2_langgraph.py -q -k sosta   # la sosta su Postgres
uv run pytest tests/test_g2_langgraph.py -q -k tetti  # i tetti e la chiusura
```

Ciascun test prova una cosa sola:

| test | cosa dimostra |
|---|---|
| `test_davanti_al_banco_...` | le quattro implementazioni sugli stessi tre casi: il grafo paga il suo ordine (triage + sintesi) e va oltre i 4 passi su saldo e segnalazione — 1/3, con `passi` violato |
| `test_il_grafo_legge_prima_i_conti_poi_il_saldo` | l'ordine della tabella: `visitati` nell'ordine giusto, i tool in fila, la sintesi che vede solo i contributi |
| `test_sotto_soglia_...` | un importo piccolo non passa dalla sosta: il tool scrive, nessuno è coinvolto |
| `test_sopra_soglia_...` | la sosta: `interrupt()` con gli argomenti validati, `Command(resume=...)` sullo stesso thread, il tool scrive solo dopo il sì |
| `test_una_respinta_...` | zero alert, e il rifiuto (chi e perché) sta nei contributi e nella risposta |
| `test_la_sosta_sopravvive_...` | Postgres: il primo checkpointer si chiude, un secondo sullo stesso database riprende il thread dov'era — e solo la ripresa paga il modello |
| `test_la_tabella_...` | `PERCORSI` è un insieme chiuso: ogni percorso comincia da uno specialista, `segnalazione` passa dalla sosta, e da ogni nodo gli archi sono quelli dichiarati |
| `test_i_tetti_vengono_prima_...` | `instrada`: budget o passi sforati → `chiusura` prima di qualunque scelta; senza proposta, `chiedi` e `scrivi` si saltano |
| `test_la_chiusura_...` | `chiudi` compone quello che c'è e non può fallire; se sforati tutti e due i tetti, vince il costo |
| `test_sopra_il_budget_...` | un triage da 600.000 token da solo supera il budget: il grafo non chiama nessun altro e chiude |
| `test_il_thread_...` | `thread_per`: username + uuid nuovo a ogni richiesta |
| `test_le_tabelle_...` | `fuori_dalle_migration`: le quattro tabelle del checkpointer stanno fuori, le altre dentro |

### 7.3 La sosta vera, col database acceso

`test_la_sosta_sopravvive_a_un_checkpointer_nuovo` è la prova grossa: apre il saver su
`lipari_ai_test`, porta il grafo fino all'interrupt, **chiude il saver**, ne apre uno nuovo sul
stesso database e riprende. Se la sosta stesse in memoria, qui fallirebbe. Su Windows gira in un
`SelectorEventLoop` dedicato, per psycopg.

### 7.4 Le misure (chiamano il modello vero: costano)

```bash
uv run python -m scripts.ingest_docs        # una volta: i documenti nell'indice
uv run python -m evals.runner               # l'harness, col cancello delle soglie
uv run python -m evals.confronto            # le quattro implementazioni sulle stesse traiettorie
```

### 7.5 Il disegno

```bash
uv run python -m scripts.disegna_grafo      # scrive docs/grafo.md dal grafo compilato
```

### 7.6 Il servizio (ancora sul ciclo a mano)

```bash
uv run uvicorn src.main:app --port 8000
```

Gli endpoint di `/api/ai/agent` girano sul ciclo scritto a mano, non sul grafo: li si prova
come sempre, con gli utenti di `scripts/seed_users.py` (password `bootcamp`):

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login \
  -d "username=mbianchi&password=bootcamp" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -X POST localhost:8000/api/ai/agent -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"message": "qual è il saldo del conto principale?"}'
```

## 8. Mappa dei file

| file | cos'è |
|---|---|
| `src/lg/stato.py` | i campi del grafo, divisi per reducer, e i tipi del contributo e della decisione |
| `src/lg/grafo.py` | `PERCORSI`, i nodi, `instrada`, `chiudi`, `costruisci_grafo`, il checkpointer, il filtro di Alembic |
| `src/agents/supervisor.py` | prompt degli specialisti e `MODELLO_TRIAGE`: il grafo li riusa, non li duplica |
| `src/agents/loop.py` | il ciclo a mano (resta: è il confronto, e tiene i prezzi e i messaggi dei tetti) |
| `alembic/env.py` | `include_object=fuori_dalle_migration`: le tabelle del checkpointer stanno fuori |
| `evals/implementazioni.py` | la quarta implementazione, `Grafo` |
| `evals/confronto.py` | i sei controlli, il quarto blocco di `main()` |
| `tests/test_g2_langgraph.py` | i dodici test del grafo, sopra descritti |
| `scripts/disegna_grafo.py` | il disegno generato dal codice |
| `docs/grafo.md` | il Mermaid + la tabella dei percorsi, generati |
