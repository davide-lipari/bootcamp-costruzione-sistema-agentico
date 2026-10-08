# Spiegazione del tetto sulle ricerche: il giro di una chiamata

Da una domanda dell'utente fino alla conclusione, passando dalle classi. Due casi: uno **buono** (traj-006, risposta trovata) e uno **senza risposta** (traj-009, la ricerca non conclude).

---

## 0. Il punto d'ingresso

Ci sono due porte sullo stesso motore, e il tetto sta solo in una:

| dove | chi gira | tetti presenti |
|---|---|---|
| `POST /api/ai/agent` → `src/api/agent.py:72` | `run_agent()` di `src/agents/loop.py:50` (ciclo scritto a mano) | `max_steps=6`, `budget=0.05 EUR`, approvazione |
| `evals/` → `AgentePrebuilt.rispondi` (`evals/implementazioni.py:115`) | `costruisci_agente()` di `src/lc/agente.py:28` (LangGraph + middleware) | `ModelCallLimitMiddleware(max_steps)`, **`ToolCallLimitMiddleware(search_documents, RICERCHE_MAX=3, "continue")`**, `HumanInTheLoopMiddleware` |

Le due porte condividono lo **stesso tool** `search_documents` (`src/agents/tools.py:89`): il tetto numero di ricerche esiste finora solo nella porta di destra, quella usata dalle misure. Quella di sinistra limita i *passi*, non le ricerche.

Il giro qui sotto è la porta destra (`AgentePrebuilt`), quella che il tetto ha cambiato.

---

## 1. Il tracciamento dei passaggi

```
main()                                  evals/confronto.py:112
  └─ confronta(path, impls, embedder, openai)          evals/confronto.py:106
       └─ misura(path, impl, embedder, openai)         evals/confronto.py:58   per ogni riga
            ├─ await asyncio.to_thread(leggi, path) → 9 casi   evals/runner.py
            ├─ utente = UserContext(**caso["input"]["user"])
            ├─ async with AsyncSessionLocal() as s:
            │    └─ deps = crea_deps(s, embedder=embedder, openai=openai)
            ├─ esito = await impl.rispondi(messaggio, utente, deps)
            │    └─ AgentePrebuilt.rispondi           evals/implementazioni.py:115
            │         ├─ build_tools_for(utente, deps)       → lista Tool registrati
            │         ├─ costruisci_agente(llm, tools, InMemorySaver(), max_steps=8)
            │         │     └─ in_fila(tools) avviene DENTRO costruisci_agente  src/lc/agente.py:56
            │         └─ ainvoke({"messages":[HumanMessage(domanda)]}, thread_id=uuid4())
            ├─ controlla(caso["expected"], esito)       evals/confronto.py:41  ← i 6 controlli
            └─ Metriche(...) → rapporto()               evals/runner.py  (soglia 0.80)
```

Dentro il grafo, il tool passa da tre trasformazioni:

```
Tool (src/agents/registry.py)          — argomenti tipizzati + ACL
  └─ in_fila(tools)                    src/lc/strumenti.py:49  mette ogni tool in coda (asyncio.Lock)
       └─ a_langchain(tool, fila)      src/lc/strumenti.py:23  → StructuredTool
            └─ chiama(**kwargs)        src/lc/strumenti.py:32  model_validate → tool.run(args)
                 └─ documenti(a)       src/agents/tools.py:89
                      ├─ embedder.embed_one(query)
                      ├─ retrieval.search_for_user(vettore, role, top_k=3)
                      │    └─ SQL: visibility = ANY(livelli)   ACL (public/internal)
                      └─ "[{document_id}] {content[:300]}"     ← il NOME sopravvive, il testo no
```

---

## 2. Caso buono — traj-006, «Quanto costa un bonifico di 26.800 euro verso Panama?»

Domanda **multi-hop**: Panama non contiene la tariffa; la tariffa è solo in `commissioni_bonifico`.

| # | classe/oggetto | cosa succede |
|---|---|---|
| 1 | `create_agent` (LangGraph) | primo `AIMessage` con `tool_calls=[search_documents("Panama bonifico")]` |
| 2 | `ToolCallLimitMiddleware.after_model` (`src/lc/agente.py:43`) | conta 1 ≤ `RICERCHE_MAX=3` → **lascia passare** |
| 3 | tool node | `chiama` → `documenti` → embedder → `search_for_user` → chunk di `panama_operativita.md` |
| 4 | `ToolMessage` | `content = "[panama_operativita] …Il costo si legge in commissioni_bonifico…"` (nome entro 300) |
| 5 | secondo `AIMessage` | il modello intuisce l'altra voce → `tool_calls=[search_documents("commissioni bonifico")]` |
| 6 | `after_model` | conta 2 ≤ 3 → **passa** |
| 7 | `ToolMessage` | chunk di `commissioni_bonifico.md`: `€15.00 + 0.20%` |
| 8 | terzo `AIMessage` | calcola 15,00 + 0,20% × 26.800 = **68,60** → **nessun `tool_calls`** |
| 9 | `come_e_finita(stato)` (`src/lc/agente.py:69`) | ultimo `AIMessage` ha `usage_metadata` e zero tool_calls → `("model", testo)` |
| 10 | `AgentePrebuilt` (`evals/implementazioni.py:128`) | `ricerche = count(ToolMessage con name="search_documents" e status="success")` = **2** |
| 11 | `controlla` (`evals/confronto.py:41`) | `tool_richiesti` ✓, `tool_vietati` ✓, `passi` 3 ≤ 5 ✓, `terminazione == "model"` ✓, `citazione` — , **`risposta_contiene("68,60")`** ✓ (68,6 / 68.60 / "68,60 €" valgono tutti) → `violati = []` |

Conclusione: **`come_e_finita = "model"`, 2 ricerche, caso superato.** Con `top_k=3` nella pratica basta spesso **1** ricerche (il primo elenco contiene già entrambi i passaggi); il caso legittimo peggiore osservato è 2, per questo il tetto è 3 e non 2.

---

## 3. Caso senza risposta — traj-009, «Qual è la commissione per il deposito cauzionale di 9.000 euro?»

Nessuna voce nel corpus: l'unico modo onesto di chiudere è **smettere di cercare**.

### 3a. Con il tetto, `exit_behavior="continue"` (così gira ora)

| # | cosa succede |
|---|---|
| 1 | ricerche 1, 2, 3 → `after_model` le lascia passare; ogni `ToolMessage` è `NIENTE_NEI_DOCUMENTI: …` |
| 2 | quarta chiamata → `after_model` **blocca**: il tool non parte; al modello torna un `ToolMessage` con `status="error"` e testo *"Tool call limit exceeded. Do not call 'search_documents' again."* |
| 3 | il run **non** si chiude (`"continue"`): il modello è richiamato, legge l'errore e **risponde senza tool** |
| 4 | `come_e_finita` → `("model", "Non trovo …")`; `Esito(..., stopped_by="model", ricerche=3)` |
| 5 | `controlla`: `passi` ≤ 6 ✓, `terminazione == "model"` ✓ → il caso **passa** (è il comportamento voluto: dire "non c'è" è una risposta) |

### 3b. Stesso caso, ma la risposta non arriva proprio

Due modi in cui `come_e_finita` restituisce **`"max_steps"`**, cioè `TETTO = "Non ho completato la richiesta entro i passi previsti: la risposta non c'è."` — che **non** è una risposta:

1. **`exit_behavior="end"`** (variante provata e scartata): alla quarta chiamata il middleware fa `jump_to="end"`; l'ultima `AIMessage` è sintetica, **senza `usage_metadata`** → `come_e_finita` → `("max_steps", …)`. Test: `assert 'max_steps' == 'model'` falliva.
2. **`ModelCallLimitMiddleware(thread_limit=max_steps, "end")`**: il modello brucia i passi chiamando sempre il tool → stessa uscita.

Nel primo giro di misure, **senza il tetto**, traj-009 ha fatto **7 ricerche in 8 passi** e 0.001791 EUR, fermandosi per un pelo su `model` e fallendo solo `passi` (8 > 6). Con il tetto: **3 ricerche, 0.000680 EUR**.

---

## 4. Le uscite possibili di `come_e_finita`

| valore | condizione su `stato["messages"]` | chi lo produce |
|---|---|---|
| `"awaiting_approval"` | c'è `__interrupt__` | `HumanInTheLoopMiddleware`, montato solo sui tool con `scrive=True` (`src/lc/agente.py:50`): `search_documents` è in lettura, quindi non lo è mai |
| `"model"` | ultimo `AIMessage` con `usage_metadata` e zero `tool_calls` | il modello ha finito |
| `"max_steps"` | tutto il resto (ultima `AIMessage` sintetica/senza usage) | `ModelCallLimitMiddleware("end")` o `ToolCallLimitMiddleware("end")` |

Solo `"model"` è una risposta; `"max_steps"` è il tetto che dice *non c'è*.

---

## 5. Schema riassuntivo

```
utente → AgentRequest / riga JSONL
   → build_tools_for → costruisci_agente (3 middleware in fila)
        → AIMessage(tool_calls) ─┬─ search_documents ≤ 3 → documenti → search_for_user → ToolMessage
                                 │                      > 3 → ToolMessage ERRORE ("continue")
                                 └─ nessun tool_calls → come_e_finita → "model"  ✅
        → max_steps raggiunto   ─────────────────────→ come_e_finita → "max_steps" ❌
   → Esito(risposta, tool_calls, steps, stopped_by, cost_eur, ricerche)
   → controlla() → 6 controlli, in ordine: tool_richiesti, tool_vietati, passi, terminazione, citazione, risposta_contiene
   → Metriche → rapporto() → README «Misure»
```
