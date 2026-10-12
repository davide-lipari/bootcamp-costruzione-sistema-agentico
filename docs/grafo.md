# Il grafo del supervisor

Generato da `scripts/disegna_grafo.py` dal grafo compilato: non è un disegno a mano.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	triage(triage)
	dati_conto(dati_conto)
	policy(policy)
	prepara(prepara)
	chiedi(chiedi)
	scrivi(scrivi)
	sintesi(sintesi)
	chiusura(chiusura)
	__end__([<p>__end__</p>]):::last
	__start__ --> triage;
	chiedi -.-> chiusura;
	chiedi -.-> scrivi;
	chiedi -.-> sintesi;
	dati_conto -.-> chiusura;
	dati_conto -.-> policy;
	dati_conto -.-> sintesi;
	policy -.-> chiusura;
	policy -.-> prepara;
	policy -.-> sintesi;
	prepara -.-> chiedi;
	prepara -.-> chiusura;
	prepara -.-> sintesi;
	scrivi -.-> chiusura;
	scrivi -.-> sintesi;
	triage -.-> chiusura;
	triage -.-> dati_conto;
	triage -.-> policy;
	triage -.-> sintesi;
	chiusura --> __end__;
	sintesi --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## I percorsi

| instradamento | nodi |
|---|---|
| `dati_conto` | dati_conto |
| `policy` | policy |
| `entrambi` | dati_conto → policy |
| `segnalazione` | dati_conto → policy → prepara → chiedi → scrivi |
