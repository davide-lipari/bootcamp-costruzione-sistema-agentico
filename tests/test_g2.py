# tests/test_g2.py — il contratto di oggi, provato con richieste vere
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from src.config import carica
from src.exceptions import RateLimitError
from src.main import app

prova = APIRouter()


@prova.get("/boom")
async def boom() -> None:
    raise RuntimeError("password del database: segreta")


@prova.get("/limite")
async def limite() -> None:
    raise RateLimitError(30)


app.include_router(prova)
client = TestClient(app, raise_server_exceptions=False)
BUSTA = {"timestamp", "status", "error", "message", "path", "details"}


def test_health_e_request_id():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "UP"
    assert r.headers["X-Request-Id"]


def test_chat_eco():
    r = client.post("/api/ai/chat", json={"session_id": "new", "message": "Ciao!"})
    assert r.status_code == 200 and r.json()["reply"] == "Echo: Ciao!"


def test_messaggio_vuoto_e_422():
    r = client.post("/api/ai/chat", json={"session_id": "s-1", "message": ""})
    assert r.status_code == 422


def test_glossario_e_404_nella_busta():
    assert client.post("/api/ai/glossary", json={"term": "IBAN"}).json()["source"]
    r = client.post("/api/ai/glossary", json={"term": "swift"})
    assert r.status_code == 404 and set(r.json()) == BUSTA and r.json()["error"] == "NOT_FOUND"
    # anche gli errori del framework: una rotta che non c'è, un metodo sbagliato
    assert set(client.get("/non-esiste").json()) == BUSTA
    sbagliato = client.get("/api/ai/glossary")
    assert sbagliato.status_code == 405 and sbagliato.json()["error"] == "METHOD_NOT_ALLOWED"


def test_500_senza_dettagli_interni():
    r = client.get("/boom")
    assert r.status_code == 500 and set(r.json()) == BUSTA
    assert "segreta" not in r.text and "Traceback" not in r.text
    assert r.headers["X-Request-Id"]  # il 500 è proprio la risposta da ritrovare nei log


def test_429_con_retry_after():
    r = client.get("/limite")
    assert r.status_code == 429 and r.headers["Retry-After"] == "30"


def test_cors_solo_per_le_origini_ammesse(monkeypatch):
    ok = client.options(
        "/api/ai/chat",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    no = client.options(
        "/api/ai/chat",
        headers={
            "Origin": "https://pagina-ostile.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in no.headers
    # e un elenco scritto senza JSON ferma l'avvio con un messaggio, non con un traceback
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    with pytest.raises(SystemExit, match="CORS_ORIGINS: valore non leggibile"):
        carica()


# ------------------------------------------------------------------ la categorizzazione
def categorizza(descrizione: str, **altro: object):
    return client.post(
        "/api/ai/categorize", json={"description": descrizione, "amount": 87.4, **altro}
    )


def test_una_bolletta_e_riconosciuta_e_dichiara_la_confidenza():
    r = categorizza("Bonifico Enel Energia")
    assert r.status_code == 200
    assert r.json()["category"] == "UTILITIES" and r.json()["subcategory"] == "ENERGY"
    assert r.json()["confidence"] == 0.8


def test_parole_intere_e_conflitto_dichiarato():
    assert categorizza("Pagamento servizi vari").json()["category"] == "OTHER"  # «eni» no
    conflitto = categorizza("Enel e Netflix, addebito unico").json()
    assert conflitto["confidence"] == 0.5 and "altra categoria" in conflitto["reasoning"]


def test_richiesta_malformata_dice_quale_campo_nella_busta():
    r = categorizza("   ", amount=-5, currency="euro")
    assert r.status_code == 422 and set(r.json()) == BUSTA
    assert r.json()["error"] == "VALIDATION_ERROR"
    campi = {d.split(":")[0] for d in r.json()["details"]}
    assert campi == {"description", "amount", "currency"}
    assert "euro" not in r.text  # il valore ricevuto non torna indietro
    non_json = client.post(
        "/api/ai/categorize", content=b"non json", headers={"Content-Type": "application/json"}
    )
    assert non_json.json()["details"] == ["corpo: JSON decode error"]


def test_il_request_id_del_chiamante_torna_uguale_quello_strano_no():
    assert (
        client.get("/health", headers={"X-Request-Id": "ops-2026-0042"}).headers["X-Request-Id"]
        == "ops-2026-0042"
    )
    strano = client.get("/health", headers={"X-Request-Id": "<script>x</script>"})
    assert strano.headers["X-Request-Id"] != "<script>x</script>"


# ------------------------------------------------------------------ l'importazione
def importa(contenuto: bytes, nome: str = "estratto.csv"):
    return client.post("/api/ai/movements/import", files={"file": (nome, contenuto, "text/csv")})


def test_ogni_riga_sbagliata_e_contata_col_suo_numero():
    csv = (
        b"date,description,amount,currency\n"  # riga 1
        b"2026-09-01,Bonifico Enel Energia,87.40,EUR\n"  # 2 buona
        b"2026-09-02,Esselunga,,EUR\n"  # 3 importo mancante
        b"2026-02-30,Conad,12.00,EUR\n"  # 4 data che non esiste
        b"01/09/2026,Coop,5.00,EUR\n"  # 5 data in un'altra forma
        b"2026-09-03,Trenitalia,39.90,euro\n"  # 6 valuta per esteso
        b"\n"  # 7 vuota: non è una riga di dati
        b'2026-09-04,Pizzeria,"12,50",EUR\n'  # 8 virgola decimale
        b"2026-09-05,Netflix,-12.99,EUR\n"  # 9 importo negativo
        b"2026-09-06,Tamoil,50.00,EUR,extra\n"  # 10 un campo in più
        b"2026-09-07,Iliad\n"  # 11 campi mancanti
        b"2026-09-08,Spotify,10.99,EUR\n"  # 12 buona
        b'2026-09-09,"Esselunga,54.10,EUR\n'  # 13 virgoletta aperta e mai chiusa
        b"2026-09-10," + b"x" * 140_000 + b",5.00,EUR\n"  # 14 un campo oltre il limite di csv
        b"2026-09-11,Coop,5.00,EUR\n"  # 15 buona: le righe dopo non spariscono
        b" 2026-09-12,Conad,8.00,EUR\n"  # 16 buona: lo spazio davanti alla data si toglie
    )
    r = importa(csv)
    assert r.status_code == 200
    assert r.json()["imported_count"] == 4
    assert [p["row"] for p in r.json()["problems"]] == [3, 4, 5, 6, 8, 9, 10, 11, 13, 14]


def test_duecento_righe_sette_sbagliate_il_conto_torna():
    righe = ["date,description,amount,currency"]
    for i in range(200):
        righe.append(f"2026-09-01,Movimento {i},{'-1' if i % 29 == 0 else '10.00'},EUR")
    r = importa("\n".join(righe).encode())
    assert r.json()["imported_count"] == 193 and len(r.json()["problems"]) == 7


def test_file_vuoto_non_csv_e_intestazione_sbagliata_mai_un_500():
    for contenuto in (b"", b"\x89PNG\r\n\x1a\n\x00\xff\xfe", b"data;descrizione;importo\n"):
        r = importa(contenuto)
        assert r.status_code == 422 and r.json()["error"] == "INVALID_FILE"


def test_senza_file_e_la_stessa_busta():
    r = client.post("/api/ai/movements/import")
    assert r.status_code == 422 and r.json()["details"] == ["file: Field required"]


def test_una_data_come_timestamp_non_passa_e_il_bom_di_excel_si():
    bom = b"\xef\xbb\xbf"  # i tre byte che Excel mette in testa a un CSV UTF-8
    r = importa(bom + b"date,description,amount,currency\n1756684800,Coop,5.00,EUR\n")
    assert r.status_code == 200  # il BOM non rompe l'intestazione
    assert r.json()["problems"] == [
        {"row": 2, "reason": "date: Value error, la data va scritta AAAA-MM-GG"}
    ]
