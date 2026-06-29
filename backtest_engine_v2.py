"""
============================================================
 ENGINE v2 — correzioni sopra backtest_engine.py
 (l'originale resta INTATTO: questo file lo importa e
  sostituisce SOLO le funzioni difettose, per confronto)
============================================================
 Correzioni:
  1) segnali_oggi_multi: segnali calcolati su SOLE candele
     COMPLETE (la candela di oggi in corso viene scartata,
     come gia' faceva il semaforo). Niente piu' segnali che
     appaiono alle 11 e spariscono alle 16; le due tabelle
     dell'app guardano finalmente la STESSA candela.
  2) niente arrotondamenti nell'engine: prezzo e stop escono
     grezzi, l'arrotondamento e' solo della UI (cosi' il
     formato a 3-4 decimali sui titoli sotto l'euro funziona).
  3) quadro_intraday accetta la data del segnale e restituisce
     None se i dati intraday NON sono successivi a quella data
     (weekend / pre-apertura -> "DATI N/D" e non un finto
     "ATTESA TRIGGER" calcolato sulla candela di ieri).
  4) i titoli scartati riportano il MOTIVO (storico corto,
     dati non scaricati, errore nel calcolo).
============================================================
"""
import pandas as pd

# tutte le funzioni e costanti originali (classifica_titolo,
# SOGLIA_NERVOSO, carica_watchlist, atr, ecc.)
from backtest_engine import *                 # noqa: F401,F403
from backtest_engine import _seg_for, atr, carica_watchlist  # espliciti

MIN_CANDELE = 250   # storico minimo per fidarsi degli indicatori (EMA200 ecc.)


def _solo_complete(df):
    """Tiene solo le candele giornaliere COMPLETE: se i dati
    includono la candela di oggi ancora in corso, la scarta."""
    oggi = pd.Timestamp.today().normalize()
    idx = df.index
    try:
        idx_norm = idx.tz_localize(None).normalize()
    except (TypeError, AttributeError):
        idx_norm = idx.normalize()
    return df[idx_norm < oggi]


# ==========================================================
# 1+2) SEGNALI DI OGGI (multi-algoritmo) su candele complete,
#      valori grezzi, motivi di scarto espliciti
# ==========================================================
def segnali_oggi_multi(tickers, periodo="2y"):
    dati = carica_watchlist(tickers, periodo)
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t and t not in dati]
    righe = []
    data_candele = None   # data della candela su cui sono calcolati i segnali

    for t, df in dati.items():
        try:
            comp = _solo_complete(df)
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico troppo corto: {len(comp)} candele)")
                continue

            prezzo = float(comp["close"].iloc[-1])
            atr_val = float(atr(comp, 14).iloc[-1])
            s = {}
            for entry in ("momentum", "pullback", "compressione"):
                try:
                    s[entry] = bool(_seg_for(entry, comp).iloc[-1])
                except Exception:
                    s[entry] = False

            d = pd.Timestamp(comp.index[-1]).date()
            if data_candele is None or d > data_candele:
                data_candele = d

            righe.append({
                "ticker": t,
                "prezzo": prezzo,                              # GREZZO: arrotonda la UI
                "stop_suggerito": prezzo - 1.5 * atr_val,      # GREZZO
                "atr": atr_val,
                "data": str(d),
                "momentum": s["momentum"], "pullback": s["pullback"],
                "compressione": s["compressione"],
                "qualcuno": any(s.values()),
            })
        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__})")

    return {"righe": righe, "falliti": falliti,
            "data_aggiornamento": str(pd.Timestamp.today().date()),
            "data_candele": str(data_candele) if data_candele else "n/d"}


# ==========================================================
# 3) QUADRO INTRADAY consapevole della data del segnale
# ==========================================================
def quadro_intraday(ticker, dopo_il=None):
    """Prezzo attuale E massimo di OGGI (5m, ~15 min di ritardo).
    Se `dopo_il` (data della candela del segnale) e' indicata e
    l'ultimo giorno intraday NON e' successivo, restituisce None:
    significa mercato chiuso / pre-apertura, e il semaforo deve
    dire DATI N/D invece di confrontare il trigger con se stesso."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period="2d", interval="5m",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        if df.empty:
            return None

        ultimo = pd.Timestamp(df.index[-1])
        ultimo_giorno = (ultimo.tz_localize(None) if ultimo.tzinfo else ultimo).normalize()

        # il giorno intraday deve essere DOPO la candela del segnale
        if dopo_il is not None:
            rif = pd.Timestamp(dopo_il)
            rif = (rif.tz_localize(None) if rif.tzinfo else rif).normalize()
            if ultimo_giorno <= rif:
                return None

        try:
            idx_norm = df.index.tz_localize(None).normalize()
        except (TypeError, AttributeError):
            idx_norm = df.index.normalize()
        oggi = df[idx_norm == idx_norm[-1]]
        if oggi.empty:
            return None
        return {"prezzo": float(oggi["close"].iloc[-1]),
                "massimo_oggi": float(oggi["high"].max()),
                "ora_dato": str(ultimo)}
    except Exception:
        return None


# ==========================================================
# SEMAFORO D'INGRESSO v2 — identico all'originale nella logica,
# ma usa il quadro_intraday corretto (passa la data del segnale)
# e riporta i motivi di scarto.
# ==========================================================
def semaforo_ingresso(tickers, periodo="2y",
                      soglia_estensione_atr=1.0, soglia_annulla_atr=0.5):
    dati = carica_watchlist(tickers, periodo)
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t and t not in dati]
    righe = []

    for t, df in dati.items():
        try:
            comp = _solo_complete(df)
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico troppo corto: {len(comp)} candele)")
                continue

            seg = {}
            for entry in ("momentum", "pullback", "compressione"):
                try:
                    seg[entry] = bool(_seg_for(entry, comp).iloc[-1])
                except Exception:
                    seg[entry] = False
            n_attivi = sum(seg.values())

            chiusura_ieri = float(comp["close"].iloc[-1])
            trigger = float(comp["high"].iloc[-1])      # massimo di ieri
            atr_ieri = float(atr(comp, 14).iloc[-1])
            data_segnale = pd.Timestamp(comp.index[-1])

            # intraday SOLO se c'e' un segnale, e SOLO se successivo
            # alla candela del segnale (altrimenti None -> DATI N/D)
            q = quadro_intraday(t, dopo_il=data_segnale) if n_attivi else None
            prezzo_ora = q["prezzo"] if q else chiusura_ieri
            massimo_oggi = q["massimo_oggi"] if q else None
            ora_dato = q["ora_dato"] if q else "n/d"

            dist_trigger_atr = ((prezzo_ora - trigger) / atr_ieri) if atr_ieri else 0.0
            dist_chiusura_atr = ((prezzo_ora - chiusura_ieri) / atr_ieri) if atr_ieri else 0.0

            if n_attivi == 0:
                stato = "—"
            elif q is None:
                stato = "DATI N/D"
            elif dist_chiusura_atr < -soglia_annulla_atr:
                stato = "ANNULLATO"
            elif massimo_oggi <= trigger:
                stato = "ATTESA TRIGGER"
            elif prezzo_ora > trigger + soglia_estensione_atr * atr_ieri:
                stato = "NON INSEGUIRE"
            elif prezzo_ora < trigger:
                stato = "RIENTRATO"
            else:
                stato = "VIA LIBERA"

            righe.append({
                "ticker": t,
                "momentum": seg["momentum"], "pullback": seg["pullback"],
                "compressione": seg["compressione"],
                "n_attivi": n_attivi, "confluenza": n_attivi >= 2,
                "qualcuno": n_attivi >= 1,
                "data_segnale": str(data_segnale.date()),
                "chiusura_ieri": chiusura_ieri,
                "trigger": trigger,
                "prezzo_ora": prezzo_ora,
                "massimo_oggi": massimo_oggi,
                "dist_trigger_atr": dist_trigger_atr,
                "dist_chiusura_atr": dist_chiusura_atr,
                "atr_ieri": atr_ieri,
                "stop_indicativo": (max(prezzo_ora, trigger) - 1.5 * atr_ieri),
                "ora_dato": ora_dato,
                "stato": stato,
            })
        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__})")

    return {"righe": righe, "falliti": falliti,
            "ora_controllo": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "soglia_estensione_atr": soglia_estensione_atr,
            "soglia_annulla_atr": soglia_annulla_atr}
