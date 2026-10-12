# Il tetto sulle ricerche: cosa fa, perché esiste, come si spiega

## La risposta in una frase

> Il programma mette un **tetto di 3 ricerche** al documento per ogni domanda dell'utente:
> se l'agente ha già consultato 3 volte il database documentale e non ha trovato la risposta,
> la quarta chiamata viene **bloccata**, il modello è costretto ad arrendersi e risponde
> "non trovo la risposta" invece di continuare a cercare all'infinito spendendo soldi.

Questo file spiega **il giro completo di una chiamata**: da quando l'utente fa una domanda
fino a quando l'agente produce la risposta (o ammette di non averla), passando per tutte
le classi coinvolte. Servono due esempi concreti:

| Esempio | Domanda | Esito |
|---|---|---|
| **traj-006** — il caso *buono* | «Quanto costa un bonifico di 26.800 euro verso Panama?» | La risposta **c'è** nei documenti: l'agente la trova |
| **traj-009** — il caso *senza risposta* | «Qual è la commissione per il deposito cauzionale di 9.000 euro?» | La risposta **non c'è** in nessun documento: qui si vede il tetto al lavoro |

---

## 1. Il problema che il tetto risolve

Immagina uno studente a un esame orale con una domanda a cui il professore *non ha risposta*.

- **Senza tetto**: lo studente continua a sfogliare gli appunti all'infinito, gira le pagine,
  cerca con parole sempre diverse, e alla fine il tempo (e il budget) è finito senza rispondere.
- **Con tetto**: dopo 3 tentativi lo studente posa gli appunti e dice «questa non la so».
  È una risposta **onesta e veloce**.

Nel codice, «sfogliare gli appunti» = chiamare il tool `search_documents`
(una ricerca vettoriale nei documenti del banco dati). Il tetto è quindi
**`RICERCHE_MAX = 3`**: massimo 3 chiamate a `search_documents` per singola conversazione.

I numeri del caso reale (misurati, vedere README «Misure»):

| | Ricerche | Costo | Esito |
|---|---|---|---|
| traj-009 **senza** tetto | 7 | 0.001791 EUR | fallisce il controllo `passi` (8 > 6) |
| traj-009 **con** tetto | **3** | **0.000680 EUR** | **passa tutti i controlli** |

---

## 2. Parole chiave (glossario per chi spiega)

Prima di entrare nel codice, cinque termini che ricorrono spesso:

| Termine | Significato umano |
|---|---|
| **Tool** | Una funzione che il modello può chiamare (es. `search_documents`). Il modello non esegue codice: *propone* una chiamata, il programma la esegue e rinfodella il risultato. |
| **AIMessage** | Un messaggio prodotto dal modello. Se contiene `tool_calls`, significa «voglio chiamare questo tool»; se non ne contiene, significa «rispondo all'utente». |
| **ToolMessage** | Il *risultato* rimandato al modello dopo una chiamata a tool (es. l'elenco dei documenti trovati). |
| **Middleware** | Un "collant" inserito tra il modello e i tool, che **osserva e intercetta** ogni passaggio. Qui i middleware contano i passi, contano le ricerche e chiedono l'approvazione umana per i tool pericolosi. |
| **`come_e_finita`** | La funzione che, guardate le messaggi finali, decide *come* è finita la conversazione: risposta del modello, tetto raggiunto, o in attesa di approvazione. |

---

## 3. Il punto d'ingresso: due porte sullo stesso motore

Il progetto ha **due modi** per far girare lo stesso agente, e il tetto esiste solo in uno:

| | Porta di sinistra | Porta di destra ← *quella spiegata qui* |
|---|---|---|
| **Dove** | `POST /api/ai/agent` → `src/api/agent.py:72` | `evals/` → `AgentePrebuilt.rispondi` (`evals/implementazioni.py:115`) |
| **Chi gira il ciclo** | `run_agent()` — ciclo scritto a mano (`src/agents/loop.py:50`) | `costruisci_agente()` — LangGraph + middleware (`src/lc/agente.py:28`) |
| **Limiti presenti** | `max_steps=6`, `budget=0.05 EUR`, approvazione | `ModelCallLimitMiddleware(max_steps)`, **`ToolCallLimitMiddleware(RICERCHE_MAX=3, "continue")`**, `HumanInTheLoopMiddleware` |
| **Il tetto alle ricerche?** | ❌ No: limita i *passi*, non le ricerche | ✅ **Sì** |

Le due porte condividono lo **stesso tool** `search_documents` (`src/agents/tools.py:89`):
è un unico motore di ricerca, con due abitanti diversi. Il tetto è stato aggiunto solo
nella porta di destra, quella **usata dalle misure** (gli esperimenti del README).

Tutto ciò che segue descrive la porta destra.

---

## 4. Il tracciamento: da `main()` fino al database

### 4a. La catena di chiamate esterna

```
main()                                       evals/confronto.py:112
  └─ confronta(path, impls, embedder, openai) evals/confronto.py:106
       └─ misura(path, impl, embedder, openai) evals/confronto.py:58   ← una volta per ogni riga del file
            ├─ await asyncio.to_thread(leggi, path) → 9 casi   evals/runner.py      ← legge le 9 domande di test
            ├─ utente = UserContext(**caso["input"]["user"])                         ← chi fa la domanda (ruolo, identità)
            ├─ async with AsyncSessionLocal() as s:
            │    └─ deps = crea_deps(s, embedder=embedder, openai=openai)            ← connessione DB + modelli
            ├─ esito = await impl.rispondi(messaggio, utente, deps)                  ← QUI GIRA L'AGENTE
            │    └─ AgentePrebuilt.rispondi            evals/implementazioni.py:115
            │         ├─ build_tools_for(utente, deps)      → lista di tool "alla misura" dell'utente
            │         ├─ costruisci_agente(llm, tools, InMemorySaver(), max_steps=8)
            │         │     └─ in_fila(tools)  avviene DENTRO costruisci_agente      src/lc/agente.py:56
            │         └─ ainvoke({"messages":[HumanMessage(domanda)]}, thread_id=uuid4())
            ├─ controlla(caso["expected"], esito)        evals/confronto.py:41   ← i 6 controlli di verifica
            └─ Metriche(...) → rapporto()                evals/runner.py         ← soglia 0.80 (80% dei casi)
```

In italiano: per ognuna delle 9 domande di test, il programma costruisce l'agente,
gli fa rispondere, poi **controlla** che la risposta rispetti 6 regole e infine
calcola il punteggio complessivo pubblicato nel README.

### 4b. Le tre "trasformazioni" che il tool subisce dentro il grafo

Prima di essere eseguito, ogni tool passa da tre mani:

```
Tool (src/agents/registry.py)           — definizione originale: argomenti tipizzati + permessi (ACL)
  └─ in_fila(tools)                     src/lc/strumenti.py:49   ← mette tutti i tool in coda (asyncio.Lock):
                                                                  un solo tool alla volta, perché condividono la sessione DB
       └─ a_langchain(tool, fila)       src/lc/strumenti.py:23   ← lo trasforma in uno StructuredTool di LangGraph
            └─ chiama(**kwargs)         src/lc/strumenti.py:32   ← valida gli argomenti, poi esegue tool.run(args)
                                                             (se scatta un'eccezione, non uccide l'agente:
                                                              rimanda al modello un messaggio di errore)
                 └─ documenti(a)        src/agents/tools.py:89   ← il vero motore di ricerca
                      ├─ embedder.embed_one(query)               ← traduce la domanda in un vettore numerico
                      ├─ retrieval.search_for_user(vettore, role, top_k=3)
                      │    └─ SQL con visibility = ANY(livelli)   ← ACL: vedi solo i documenti che il tuo ruolo può vedere
                      └─ "[{document_id}] {content[:300]}"       ← output: id + primi 300 caratteri.
                                                                   L'ID sopravvive (serve per la citazione), il testo è troncato
```

**Perché `in_fila` e il lock?** Tutti i tool dello stesso run condividono una singola
sessione di database: due operazioni contemporanee la farebbero esplodere. La coda
garantisce «uno alla volta».

---

## 5. Il cuore: i tre middleware dentro `costruisci_agente`

`costruisci_agente` (`src/lc/agente.py:28-60`) monta **tre middleware in fila**.
Sono le regole del gioco, in ordine di passaggio:

| # | Middleware | Cosa fa | Parametri chiave |
|---|---|---|---|
| 1 | `ModelCallLimitMiddleware` | Conta le **richieste al modello** (i "passi") | `thread_limit=max_steps`, `exit_behavior="end"` |
| 2 | `ToolCallLimitMiddleware` | Conta le chiamate a **`search_documents`** ← **questo è il tetto** | `thread_limit=RICERCHE_MAX=3`, `exit_behavior="continue"` |
| 3 | `HumanInTheLoopMiddleware` | Ferma l'agente e chiede **approvazione umana** per i tool che *scrivono* | montato solo sui tool con `scrive=True` |

Due dettagli che vale la pena sapere per spiegarli bene:

- **`thread_limit` e non `run_limit`**: dopo un'interruzione (approvazione umana) la ripresa
  è una nuova "esecuzione"; con `run_limit` il contatore ripartirebbe da zero e il tetto
  salterebbe. Con `thread_limit` il conto dura per tutta la conversazione.
- **`exit_behavior="continue"` e non `"end"`**: è la scelta che tiene in vita il caso
  traj-009. Con `"end"` la conversazione si chiuderebbe di colpo e la risposta risulterebbe
  `max_steps` (cioè «non completato»). Con `"continue"` la conversazione **continua**:
  il modello legge l'errore e risponde con calma. La differenza si vede al §7.

E c'è un quarto middleware «implicito» che però è una funzione: `come_e_finita` (`src/lc/agente.py:69`),
di cui si parla all'§8.

---

## 6. Caso buono — traj-006: la risposta c'è

**Domanda**: «Quanto costa un bonifico di 26.800 euro verso Panama?»

Perché è un caso *multi-hop* (a più salti): il documento su Panama **non contiene** la tariffa,
dice solo *dove* cercarla (`commissioni_bonifico`). L'agente deve fare **due** ricerche
in sequenza, capendo da solo il secondo passo.

| # | Chi agisce | Cosa succede |
|---|---|---|
| 1 | `create_agent` (LangGraph) | Il modello emette il primo `AIMessage` con `tool_calls=[search_documents("Panama bonifico")]` |
| 2 | `ToolCallLimitMiddleware.after_model` (`src/lc/agente.py:43`) | Conta: 1 ≤ 3 → **lascia passare** |
| 3 | Tool node | `chiama` → `documenti` → embedder → `search_for_user` → ritorna un chunk di `panama_operativita.md` |
| 4 | `ToolMessage` | Il modello riceve: `"[panama_operativita] …Il costo si legge in commissioni_bonifico…"` (entro 300 caratteri) |
| 5 | secondo `AIMessage` | Il modello capisce il rimando → `tool_calls=[search_documents("commissioni bonifico")]` |
| 6 | `after_model` | Conta: 2 ≤ 3 → **passa** |
| 7 | `ToolMessage` | Chunk di `commissioni_bonifico.md`: `€15.00 + 0.20%` |
| 8 | terzo `AIMessage` | Calcola 15,00 + 0,20% × 26.800 = **68,60** → **nessun `tool_calls`**: ha finito |
| 9 | `come_e_finita(stato)` (`src/lc/agente.py:69`) | L'ultimo `AIMessage` ha `usage_metadata` e zero tool_calls → `("model", testo)` |
| 10 | `AgentePrebuilt` (`evals/implementazioni.py:128`) | `ricerche` = conta i `ToolMessage` con `name="search_documents"` e `status="success"` → **2** |
| 11 | `controlla` (`evals/confronto.py:41`) | `tool_richiesti` ✓, `tool_vietati` ✓, `passi` 3 ≤ 5 ✓, `terminazione == "model"` ✓, `citazione` —, **`risposta_contiene("68,60")`** ✓ (valgono anche `68,6` / `68.60` / `"68,60 €"`) → `violati = []` |

**Conclusione**: `come_e_finita = "model"`, 2 ricerche, caso superato.

**Perché il tetto è 3 e non 2?** Con `top_k=3` (3 chunk restituiti per ricerca) nella pratica
basta spesso **1** ricerca, perché il primo elenco contiene già entrambi i passaggi.
Il caso peggiore *legittimo* osservato richiede **2** ricerche. Il tetto di 3 lascia
quindi un margine di sicurezza senza mai essere raggiunto nei casi normali:
nei casi buoni il tetto **non si nota nemmeno**.

---

## 7. Caso senza risposta — traj-009: qui il tetto serve

**Domanda**: «Qual è la commissione per il deposito cauzionale di 9.000 euro?»

Nessun documento del corpus ne parla. L'unico modo *onesto* di chiudere è
**smettere di cercare** e dirlo.

### 7a. Con il tetto, `exit_behavior="continue"` (così gira ora)

| # | Cosa succede |
|---|---|
| 1 | Ricerche 1, 2, 3 → `after_model` le lascia passare; ogni `ToolMessage` dice `NIENTE_NEI_DOCUMENTI: …` |
| 2 | **Quarta chiamata** → `after_model` **blocca**: il tool non parte. Al modello torna un `ToolMessage` con `status="error"` e testo *"Tool call limit exceeded. Do not call 'search_documents' again."* |
| 3 | Il run **non** si chiude (è la scelta `"continue"`): il modello viene richiamato, legge l'errore e **risponde senza tool** |
| 4 | `come_e_finita` → `("model", "Non trovo …")`; `Esito(..., stopped_by="model", ricerche=3)` |
| 5 | `controlla`: `passi` ≤ 6 ✓, `terminazione == "model"` ✓ → il caso **passa**. È il comportamento voluto: dire «non c'è» **è** una risposta |

Nota: `ricerche = 3`, non 4, perché la quarta chiamata è stata bloccata e produce un
`ToolMessage` con `status="error"`, che il contatore di `evals/implementazioni.py:128`
non include (conta solo le ricerche *eseguite*).

### 7b. Stesso caso, ma la risposta non arriva proprio

Due modi in cui `come_e_finita` restituisce **`"max_steps"`**, cioè
`TETTO = "Non ho completato la richiesta entro i passi previsti: la risposta non c'è."`
— che **non** è una risposta vera:

1. **`exit_behavior="end"`** (variante provata e scartata): alla quarta chiamata il
   middleware fa `jump_to="end"`; l'ultima `AIMessage` è sintetica, **senza
   `usage_metadata`**, quindi `come_e_finita` → `("max_steps", …)`. Il test
   `assert 'max_steps' == 'model'` falliva.
2. **`ModelCallLimitMiddleware(thread_limit=max_steps, "end")`**: il modello spreca tutti
   i passi chiamando sempre il tool → stessa uscita brutale.

Ecco perché la variante scelta è `"continue"`: **lascia al modello l'ultima parola**,
così la conversazione finisce con una risposta umana e non con un muro.

Nel primo giro di misure, **senza il tetto**, traj-009 aveva fatto **7 ricerche in 8 passi**
e speso 0.001791 EUR, fermandosi per un pelo su `model` ma fallendo il controllo `passi`
(8 > 6). **Con il tetto**: **3 ricerche, 0.000680 EUR**, tutti i controlli superati.

---

## 8. Le tre uscite possibili di `come_e_finita`

`come_e_finita(stato)` (`src/lc/agente.py:69-77`) guarda i messaggi finali e decide
come è finita la partita:

| Valore | Condizione su `stato["messages"]` | Chi lo produce |
|---|---|---|
| `"awaiting_approval"` | C'è `__interrupt__` | `HumanInTheLoopMiddleware`, montato solo sui tool con `scrive=True` (`src/lc/agente.py:50`). `search_documents` è in lettura, quindi **non lo genera mai** |
| `"model"` | Ultimo `AIMessage` con `usage_metadata` e zero `tool_calls` | Il modello ha finito **da solo**: è l'unica uscita che è una risposta |
| `"max_steps"` | tutto il resto (ultima `AIMessage` sintetica / senza usage) | `ModelCallLimitMiddleware("end")` o `ToolCallLimitMiddleware("end")` |

**Regola da ricordare**: solo `"model"` è una risposta; `"max_steps"` è il tetto che dice
*non c'è*. Per questo il `ToolCallLimitMiddleware` è configurato su `"continue"`:
vuole che l'uscita sia `"model"`, non `"max_steps"`.

---

## 9. Schema riassuntivo (tutto su una lavagna)

```
utente → AgentRequest / riga JSONL
   → build_tools_for → costruisci_agente (3 middleware in fila)
        → AIMessage(tool_calls) ─┬─ search_documents ≤ 3 → documenti → search_for_user → ToolMessage
                                 │                      > 3 → ToolMessage ERRORE (exit_behavior "continue")
                                 └─ nessun tool_calls → come_e_finita → "model"  ✅
        → max_steps raggiunto   ─────────────────────→ come_e_finita → "max_steps" ❌
   → Esito(risposta, tool_calls, steps, stopped_by, cost_eur, ricerche)
   → controlla() → 6 controlli, in ordine: tool_richiesti, tool_vietati, passi,
                     terminazione, citazione, risposta_contiene
   → Metriche → rapporto() → README «Misure»
```

---

## 10. Come si spiega in 30 secondi (scaletta per esporre)

1. **Il problema**: un agente senza limiti, di fronte a una domanda senza risposta nei
   documenti, cerca e cerca: spreca passi, soldi, e alla fine non risponde comunque.
2. **La soluzione**: un middleware (`ToolCallLimitMiddleware`) conta le chiamate a
   `search_documents` e ne ammette al massimo **3 per conversazione**.
3. **La scelta furba**: quando il tetto scatta, il run **non** viene chiuso forzatamente
   (`exit_behavior="continue"`): al modello arriva un messaggio di errore gentile
   («non chiamarlo più»), il modello capisce e risponde «non trovo la risposta».
   Così l'uscita è `"model"` (una risposta vera), non `"max_steps"` (un muro).
4. **La prova**: senza tetto, il caso traj-009 faceva 7 ricerche e falliva; con il tetto
   fa 3 ricerche, costa meno della metà e passa tutti i controlli. Nei casi normali
   (traj-006) il tetto non si nota: bastano 1-2 ricerche su un massimo di 3.
