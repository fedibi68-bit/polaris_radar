# 🌟 POLARIS Radar — Guida al Deploy su Streamlit Cloud

Questa guida ti permette di accedere a POLARIS Radar da **smartphone**
tramite browser, senza installare nulla sul telefono.

---

## 📁 File da caricare nel repo GitHub

Metti TUTTI questi file nella stessa cartella prima di procedere:

```
📁 polaris-radar/                  ← nome cartella a tua scelta
   ├── polaris_radar_v1.py         ← app principale
   ├── backtest_engine.py          ← ENGINE (dalla tua cartella locale)
   ├── backtest_engine_v3.py       ← ENGINE V3 (dalla tua cartella locale)
   ├── titoli_radar.json           ← watchlist (quella inclusa qui o la tua)
   ├── requirements.txt            ← dipendenze Python
   ├── .gitignore                  ← file da escludere
   └── .streamlit/
       └── config.toml             ← tema e configurazione
```

> ⚠️ **IMPORTANTE**: `backtest_engine.py` e `backtest_engine_v3.py`
> sono file tuoi — copiaci quelli dalla cartella locale C:\Trading\

---

## PASSO 1 — Crea un account GitHub (se non ce l'hai)

1. Vai su **https://github.com**
2. Clicca **Sign up** e registrati
3. Verifica l'email

---

## PASSO 2 — Crea un repository privato

1. Clicca il **+** in alto a destra → **New repository**
2. Impostazioni:
   - **Repository name**: `polaris-radar` (o qualsiasi nome)
   - **Private** ← obbligatorio (contiene il tuo codice)
   - ✅ Add a README file (opzionale)
3. Clicca **Create repository**

---

## PASSO 3 — Carica i file

### Metodo A — Via browser (più semplice)

1. Apri il tuo repo su GitHub
2. Clicca **Add file** → **Upload files**
3. Trascina TUTTI i file elencati sopra
4. Per la cartella `.streamlit/config.toml`:
   - Clicca **Add file** → **Create new file**
   - Nel campo nome scrivi: `.streamlit/config.toml`
   - Incolla il contenuto del file config.toml
5. Clicca **Commit changes**

### Metodo B — Via Git (se hai Git installato)

```bash
cd C:\Trading\polaris-radar-deploy
git init
git remote add origin https://github.com/TUO_USERNAME/polaris-radar.git
git add .
git commit -m "primo deploy POLARIS Radar"
git push -u origin main
```

---

## PASSO 4 — Crea account Streamlit Community Cloud

1. Vai su **https://share.streamlit.io**
2. Clicca **Sign up** e accedi con il tuo account **GitHub**
   (autorizza Streamlit ad accedere ai tuoi repo)

---

## PASSO 5 — Deploya l'app

1. Su **share.streamlit.io** clicca **New app**
2. Compila i campi:
   - **Repository**: `TUO_USERNAME/polaris-radar`
   - **Branch**: `main`
   - **Main file path**: `polaris_radar_v1.py`
3. Clicca **Deploy!**
4. Aspetta 2–3 minuti (il primo deploy installa le dipendenze)
5. L'app sarà disponibile all'URL tipo:
   `https://polaris-radar-v1.streamlit.app`

---

## PASSO 6 — Accedi da smartphone

1. Apri **Safari** (iPhone) o **Chrome** (Android)
2. Vai all'URL della tua app (es. `https://polaris-radar-v1.streamlit.app`)
3. **Aggiungi alla schermata Home**:
   - iPhone: tocca **Condividi** → **Aggiungi a schermata Home**
   - Android: tocca i **tre puntini** → **Aggiungi a schermata Home**
4. L'app si apre come se fosse un'app nativa

---

## ⚠️ NOTA IMPORTANTE — Watchlist

Streamlit Cloud ha un **filesystem temporaneo**: le modifiche alla watchlist
(aggiunta/rimozione titoli) vengono perse al riavvio dell'app.

**Soluzione**:
- Nella sezione Watchlist clicca **Esporta watchlist (JSON)**
- Salva il file `titoli_radar.json` scaricato
- Caricalo nel tuo repo GitHub sostituendo il vecchio `titoli_radar.json`
- L'app al prossimo riavvio partirà con la watchlist aggiornata

In alternativa, per uso quotidiano, basta ri-aggiungere i titoli
all'avvio (vengono reclassificati automaticamente).

---

## 🔄 Come aggiornare l'app

Ogni volta che aggiorni `polaris_radar_v1.py` in locale:
1. Carica il nuovo file su GitHub (sovrascrive il vecchio)
2. Streamlit rileva la modifica e riavvia automaticamente in ~30 secondi

---

## 🔒 Sicurezza

- Il repo è **privato**: solo tu puoi vederlo
- L'URL dell'app è pubblico ma non indicizzato
- Se vuoi proteggere l'accesso, usa **Streamlit secrets** o
  imposta il repo come privato e l'app come "accessible only to viewers"
  nelle impostazioni di Streamlit Cloud

---

## ❓ Problemi comuni

| Problema | Soluzione |
|---|---|
| `ModuleNotFoundError` | Aggiungi il modulo mancante in `requirements.txt` |
| App si riavvia spesso | Normale per il piano gratuito (va in sleep dopo 7 gg di inattività) |
| Watchlist resettata | Vedi sezione "Nota Importante" sopra |
| Errore `backtest_engine` | Assicurati di aver caricato entrambi i file engine nel repo |
