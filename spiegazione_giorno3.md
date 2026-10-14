# Giorno 3 — cosa ho fatto, in parole semplici

Questo file racconta il lavoro del Giorno 3 ("aggregazione dei criteri") e come si lancia
la misura vera. Non serve essere esperti per seguirlo.

---

## 1. Cosa doveva fare il Giorno 3

Finora l'agente rispondeva bene a una domanda alla volta. Il Giorno 3 chiede di farlo
**per una giornata intera di lavoro**: cinquanta domande di fila, senza dimenticare chi è
il cliente, senza superare i limiti di token, e **con lo stile che l'operatore ha scelto**.

In pratica quattro cose:

1. **Instradamento a 5 vie** — prima di rispondere, un nodo decide se la domanda riguarda
   i conti, i movimenti, le policy, l'apertura di una segnalazione, oppure tutto insieme.
2. **Memoria** — le preferenze dell'operatore (es. "rispondi in modo breve") si salvano su
   Postgres e si ricordano al turno dopo.
3. **Stile** — la risposta finale viene scritta nello stile ricordato.
4. **Conversazione lunga** — i messaggi vecchi vengono tagliati (massimo 12 nello stato),
   ma i fatti (cliente, conti) restano.

---

## 2. I file toccati

| File | Cos'è, in una frase |
|---|---|
| `src/lg/stato.py` | Il "modulo" che i nodi si passano. Ci ho aggiunto l'istruttoria e il `Contesto` (stile, sforzo). |
| `src/lg/grafo.py` | Il grafo vero e proprio: triage a 5 vie, nodi nuovi `istruttoria` e `chiusura`, e la risposta finale scritta nello stile giusto. |
| `src/lg/memoria.py` | Salva e rilegge le preferenze dell'operatore su Postgres, in uno spazio a suo nome. |
| `src/lg/conversazione.py` | Trasforma una sessione di chat in tanti turni: ogni turno gira il grafo, tiene i fatti, taglia i messaggi vecchi. |
| `src/lg/tempo.py` | Grafi che si fermano a metà (sosta) e riprendono da dove erano. |
| `evals/durata.py` | **La misura**: cinquanta domande, e alla fine una tabellina con messaggi nello stato, cliente, costo — e quante risposte nomina il cliente (la "sonda"). |
| `tests/test_g3_dura.py` | I 17 test che fanno da cancello: se passano, il Giorno 3 è a posto. |
| `tests/test_g2_langgraph.py` | Aggiunti due aiutini (`copione_segnalazione`, `alert`) che i test nuovi usano. |

I file di ieri non li ho rifatti: ho toccato solo `stato.py` e `grafo.py` (quelli che il
nuovo grafo cambia) e aggiunto i due aiutini nei test del Giorno 2.

---

## 3. Come ho verificato che funziona

- `uv run ruff check` → pulito
- `uv run mypy src/lg evals/durata.py` → nessun problema
- `uv run pytest -q` → **172 test verdi** (20 dei quali nuovi: 17 del Giorno 3, 3 dell'estensione B)

Questo è il vero "cancello": se i test passano, il codice è a posto.

---

## 4. Perché la misura vera non l'ho potuta finire

La misura vera (`uv run python -m evals.durata`) gira 50 turni sul modello vero
(`openai/gpt-oss-120b` su Groq). Con la chiave free di Groq il limite è di
**200.000 token al giorno**: li abbiamo consumati durante i tentativi, e il limite si
riapre lentamente (qualche centinaio di token al minuto). Il codice aspetta e riprova,
ma con l'attuale budget non basta il tempo.

**Soluzione**: domattina il budget si riapre da solo (si resetta ogni giorno). Puoi
lanciare tu il comando qui sotto e vedere la tabellina vera.

---

## 5. Come lanciare la misura da solo

1. Assicurati che Postgres sia su (`docker start lipari-pg`, se non già avviato).
2. Nel `.env` deve esserci la chiave Groq (`OPENAI_API_KEY`) e il modello di default.
3. Lancia:

   ```bash
   uv run python -m evals.durata
   ```

4. Alla fine appare una cosa simile:

   ```
   turno | messaggi nello stato | cliente nei fatti | costo del blocco | costo totale
       10 |                   12 | C-10234           |           0.0123 | 0.0123
       20 |                   12 | C-10234           |           0.0098 | 0.0221
       ...
   sonde con il cliente nella risposta: 1 su 1
   ```

   Cosa guardare:
   - **messaggi nello stato** non deve salire oltre 12: vuol dire che il taglio funziona.
   - **cliente nei fatti** deve restare `C-10234` per tutto: vuol dire che non dimentica.
   - **sonde ... N su 1**: se è 1 su 1, la risposta nomina il cliente ricordato.
   - **costo del blocco** deve smettere di salire: se continua a crescere, c'è qualcosa
     che si ripete per sbaglio.

Se il comando muore per il limite di Groq, riprova più tardi: non è un bug del codice.

---

## 6. La prova che ho fatto in locale (Ollama)

Per non restare a mani vuote ho fatto una **prova ridotta** (6 turni invece di 50) con
il tuo Ollama e `llama3.1:8b`, girando in CPU. Risultato:

```
turno | messaggi nello stato | cliente nei fatti | costo del blocco | costo totale
    3 |                    6 | C-10234           |           0.0000 | 0.0000
    6 |                   12 | C-10234           |           0.0000 | 0.0000
sonde con il cliente nella risposta: 1 su 1
```

In pratica: tutto il percorso (instradamento, strumenti, memoria, taglio messaggi) funziona
dal primo all'ultimo turno, in circa 8 minuti. La sonda passa: la risposta nomina il cliente
e i due conti imparati in turni precedenti.

**Attenzione però**: con un modello piccolo i numeri sono solo indicativi. Un modello
scarso può fallire la sonda per stupidaggine, non perché il sistema sia rotto. Per questo
la misura seria va fatta sul modello vero (punto 5).

La prova l'ho fatta con uno script temporaneo fuori dal repository: il repo non è stato
toccato e i test sono sempre verdi.

---

## 7. L'estensione: la strada B (la preferenza a richiesta)

Del Giorno 3 ho scelto l'estensione **B**: l'operatore può cambiare le preferenze in mezzo
alla conversazione, con una frase normale, e la conversazione salva o cancella.

Ramo: `g3-strada-b` (creato dal cancello, `main` non è stato toccato).

### Come funziona

- Prima di rispondere, un nodo nuovo (`ascolta`) guarda la domanda e risponde con **una
  parola sola**: `breve`, `dettagliato`, `dimentica` oppure `nessuna`.
- Se la parola è `breve` o `dettagliato`, la preferenza si salva **in uno schema chiuso**:
  si può scegliere solo fra quei due valori. La frase dell'operatore non si copia.
- Se la parola è `dimentica`, la preferenza sparisce **subito**, non alla scadenza.
- In tutti e due casi la risposta di quel turno contiene una conferma: l'operatore vede
  che è successo qualcosa.
- Se la parola è `nessuna`, **non si scrive nulla**: una normale domanda sui conti lascia
  la memoria come sta.

### Le trappole, e come le ho evitate

| Trappola | Come è risolta |
|---|---|
| La richiesta arriva scritta come capita («rispondimi in elenco», «sii succinto»…) | Il valore che si salva **non è la frase**: è una parola fra quelle ammesse dallo schema. |
| Una preferenza detta dentro una domanda su un cliente non deve portare il cliente nella memoria | Nello schema c'è solo il campo `formato`: un cliente non ha dove entrare. C'è un test che lo verifica. |
| Il namespace (lo spazio delle preferenze di un operatore) è una decisione del codice | Non cambia: la scrittura passa sempre per `memoria.salva`/`memoria.dimentica`, che costruiscono la chiave con l'utente autenticato. |
| La cancellazione deve essere immediata e visibile | La voce sparisce dallo store nello stesso turno, e la conferma sta nella risposta. |

### Quanto costa in più

Una chiamata piccolissima in più per turno (una parola di risposta): il suo prezzo entra
nel costo del turno, come tutte le altre. Nella misura da 50 turni si somma al resto.

### I tre casi scritti (in `tests/test_g3_dura.py`)

1. `test_quando_l_operatore_chiede_una_preferenza_si_salva_e_si_vede_subito` —
   «d'ora in poi rispondimi in modo dettagliato»: nello store c'è solo
   `{"formato": "dettagliato"}`, la conferma è nella risposta, e **quel turno** risponde
   già nello stile nuovo.
2. `test_dimentica_cancella_subito_e_l_operatore_lo_ve_de` —
   «dimentica le mie preferenze»: la voce non c'è più, e l'operatore lo vede.
3. `test_una_preferenza_in_una_domanda_non_porta_il_cliente_nella_memoria` —
   «del cliente C-10234 quali conti ha? E da ora in breve»: si salva solo
   `{"formato": "breve"}`, niente `C-10234` da nessuna parte; e la domanda dopo, senza
   richieste, non scrive nulla.

### Come si prova

Tutti e tre girano con i modelli finti, quindi non costa nulla e non aspetta nessuno:

```bash
uv run pytest -q tests/test_g3_dura.py
```

---

## 8. Riepilogo in tre righe

- Codice del Giorno 3 fatto e verificato: 172 test verdi, ruff e mypy puliti.
- Estensione B fatta sul ramo `g3-strada-b`: la preferenza si salva e si cancella a
  richiesta, sempre dentro lo schema chiuso.
- La misura vera da 50 turni è pronta, ma serve il budget Groq (si riapre domattina):
  `uv run python -m evals.durata`.
