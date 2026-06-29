"""
============================================================
 ENGINE v3 — I TRE ALGORITMI SISTEMATI (per confronto con v2)
 Eredita dal v2 tutte le correzioni tecniche (candele complete,
 valori grezzi, semaforo onesto). Stessi TRE algoritmi, stesse
 colonne, ma corretti nei loro difetti:

  MOMENTUM a EVENTO
    Prima: uno "stato" (le condizioni restano vere per settimane,
    segnale fotocopia ogni giorno). Ora: segnala SOLO il primo
    giorno in cui la forza si accende (EMA20>50, RSI>50, ADX>25),
    con riarmo di 3 giorni (le condizioni devono essere state
    spente, altrimenti lo sfarfallio dell'ADX sulla soglia
    genererebbe finti "primi giorni") e volume sopra la media.

  PULLBACK in ATR
    Prima: profondita' del calo fissa all'1.5% — rumore sui
    titoli nervosi, a cui questo algoritmo e' destinato.
    Ora: il calo dal massimo si misura in ATR del titolo
    (>= 1.0 ATR): "respiro" significa la stessa cosa per tutti.

  COMPRESSIONE in SEQUENZA
    Prima: due logiche opposte sotto un OR (setup *oppure*
    breakout, anche senza compressione). Ora: UNA storia sola —
    la molla deve essersi CARICATA nei 5 giorni precedenti
    (volume calante + range stretto + RSI neutro, 2 giorni di
    fila, sopra EMA50) e OGGI deve SCATTARE verso l'alto
    (candela verde, corpo >= 60%, volume 1.5x). Niente piu'
    segnali "in formazione" senza direzione.

  + FILTRO DI REGIME (opzionale, interruttore nella UI):
    i segnali valgono solo se l'indice di riferimento del titolo
    (FTSE MIB per i .MI, S&P 500 per il resto) e' sopra la sua
    EMA200. Mercato in tendenza ribassista = tutto spento.
============================================================
"""
import numpy as np
import pandas as pd

from backtest_engine import (ema, rsi, atr, adx, macd_hist, barre_da_max,
                             carica_watchlist, carica_dati,
                             classifica_titolo, SOGLIA_NERVOSO)   # noqa: F401
from backtest_engine_v2 import _solo_complete, MIN_CANDELE, quadro_intraday

ALGOS_V3 = ("momentum", "pullback", "compressione")

PARAMS_V3 = {
    "momentum":     {"adx_min": 25, "rsi_min": 50, "riarmo": 3},
    "pullback":     {"prof_atr": 1.0, "punteggio_min": 3, "pb_barre": 4},
    "compressione": {"range_max": 1.2, "giorni_armata": 5, "corpo_min": 60, "vol_mult": 1.5},
}


# ==========================================================
# MOMENTUM a EVENTO (primo giorno di forza, con riarmo)
# ==========================================================
def seg_momentum_v3(df, adx_min=25, rsi_min=50, riarmo=3):
    c, v = df["close"], df["volume"]
    core = ((ema(c, 20) > ema(c, 50)) & (rsi(c, 14) > rsi_min) &
            (adx(df, 14) > adx_min)).fillna(False)
    spento_prima = (core.shift(1).rolling(riarmo).sum() == 0)   # spento da `riarmo` giorni
    evento = (core & spento_prima).fillna(False)
    conferma_vol = (v > v.rolling(20).mean()).fillna(False)
    return (evento & conferma_vol).fillna(False)


# ==========================================================
# PULLBACK con profondita' in ATR (non in % fissa)
# ==========================================================
def seg_pullback_v3(df, prof_atr=1.0, punteggio_min=3, pb_barre=4):
    c, o, h = df["close"], df["open"], df["high"]
    e21, e50, e200 = ema(c, 21), ema(c, 50), ema(c, 200)
    trend = (c > e200) & (e21 > e50) & (e50 > e200)
    a = atr(df, 14)
    maxrec = h.rolling(20).max()
    dist_atr = (maxrec - c) / a.replace(0, np.nan)      # profondita' in ATR
    pull = (dist_atr >= prof_atr) & (barre_da_max(h, 20) >= pb_barre)
    r, hist, ax = rsi(c, 14), macd_hist(c), adx(df, 14)
    score = (((r > 40) & (r < 75)).astype(int) + ((hist < 0) & (hist > hist.shift())).astype(int) +
             (c > o).astype(int) + ((c > e21*0.98) & (c < e21*1.03)).astype(int) + (ax > 18).astype(int))
    veto = a > a.rolling(50).mean() * 2.5
    return (trend & pull & (score >= punteggio_min) & (~veto)).fillna(False)


# ==========================================================
# COMPRESSIONE in SEQUENZA: molla carica PRIMA, scatto OGGI
# ==========================================================
def seg_compressione_v3(df, range_max=1.2, giorni_armata=5, corpo_min=60, vol_mult=1.5):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    sopra = c >= ema(c, 50)
    # --- la molla (il setup): come prima, ma e' solo il presupposto ---
    r = rsi(c, 14)
    c_vol = v.rolling(3).mean() < v.shift(3).rolling(3).mean() * 0.85
    c_range = (h - l) / c * 100 < range_max
    cnt = ((r >= 30) & (r <= 60)).astype(int) + c_vol.astype(int) + c_range.astype(int) + sopra.astype(int)
    molla = ((cnt >= 3) & (cnt >= 3).shift(1)) & sopra
    # la molla deve essersi caricata in uno degli ultimi `giorni_armata` giorni (PRIMA di oggi)
    armata = (molla.shift(1).rolling(giorni_armata).max() >= 1)
    # --- lo scatto (oggi): candela verde decisa con volume ---
    corpo = (c - o).abs() / (h - l).replace(0, np.nan) * 100
    scatto = (c > o) & (corpo >= corpo_min) & (v >= v.rolling(20).mean() * vol_mult)
    return (armata & scatto & sopra).fillna(False)


def _seg_v3(entry, df):
    p = PARAMS_V3[entry]
    if entry == "momentum":
        return seg_momentum_v3(df, **p)
    if entry == "pullback":
        return seg_pullback_v3(df, **p)
    return seg_compressione_v3(df, **p)


# ==========================================================
# FILTRO DI REGIME (indice sopra la sua EMA200)
# ==========================================================
def _bench_for(ticker):
    return "FTSEMIB.MI" if str(ticker).upper().endswith(".MI") else "^GSPC"


def carica_regimi(tickers, periodo="2y"):
    """Per ogni benchmark necessario: serie booleana close > EMA200.
    Se un benchmark non si scarica -> None (e il filtro NON blocca:
    meglio nessun filtro che un filtro a caso)."""
    out = {}
    for b in sorted({_bench_for(t) for t in tickers if t}):
        try:
            dfb = _solo_complete(carica_dati(b, periodo))
            out[b] = (dfb["close"] > ema(dfb["close"], 200)).fillna(False)
        except Exception:
            out[b] = None
    return out


def regime_al(regimi, ticker, quando):
    """True/False: benchmark sopra EMA200 a quella data? None se n/d."""
    serie = regimi.get(_bench_for(ticker))
    if serie is None or serie.empty:
        return None
    serie = serie[serie.index <= pd.Timestamp(quando)]
    return bool(serie.iloc[-1]) if len(serie) else None


# ==========================================================
# SEGNALI DI OGGI v3 (stesse 3 colonne di v1/v2 + regime)
# ==========================================================
def segnali_oggi_multi(tickers, periodo="2y", usa_regime=True):
    dati = carica_watchlist(tickers, periodo)
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t and t not in dati]
    regimi = carica_regimi(tickers, periodo) if usa_regime else {}
    righe = []
    data_candele = None

    for t, df in dati.items():
        try:
            comp = _solo_complete(df)
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico troppo corto: {len(comp)} candele)")
                continue

            grezzi = {}
            for entry in ALGOS_V3:
                try:
                    grezzi[entry] = bool(_seg_v3(entry, comp).iloc[-1])
                except Exception:
                    grezzi[entry] = False

            d = pd.Timestamp(comp.index[-1])
            reg = regime_al(regimi, t, d) if usa_regime else None
            blocco = usa_regime and (reg is False)
            s = {k: (False if blocco else v) for k, v in grezzi.items()}

            if data_candele is None or d.date() > data_candele:
                data_candele = d.date()

            prezzo = float(comp["close"].iloc[-1])
            atr_val = float(atr(comp, 14).iloc[-1])
            righe.append({
                "ticker": t,
                "prezzo": prezzo, "stop_suggerito": prezzo - 1.5 * atr_val, "atr": atr_val,
                "data": str(d.date()),
                **s,
                "regime": reg,                          # True/False/None (n/d)
                "spenti_dal_regime": sum(grezzi.values()) if blocco else 0,
                "qualcuno": any(s.values()),
            })
        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__})")

    return {"righe": righe, "falliti": falliti,
            "data_aggiornamento": str(pd.Timestamp.today().date()),
            "data_candele": str(data_candele) if data_candele else "n/d",
            "usa_regime": usa_regime}


# ==========================================================
# SEMAFORO D'INGRESSO v3 (stessa meccanica della v2)
# ==========================================================
def semaforo_ingresso(tickers, periodo="2y",
                      soglia_estensione_atr=1.0, soglia_annulla_atr=0.5,
                      usa_regime=True):
    dati = carica_watchlist(tickers, periodo)
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t and t not in dati]
    regimi = carica_regimi(tickers, periodo) if usa_regime else {}
    righe = []

    for t, df in dati.items():
        try:
            comp = _solo_complete(df)
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico troppo corto: {len(comp)} candele)")
                continue

            grezzi = {}
            for entry in ALGOS_V3:
                try:
                    grezzi[entry] = bool(_seg_v3(entry, comp).iloc[-1])
                except Exception:
                    grezzi[entry] = False

            data_segnale = pd.Timestamp(comp.index[-1])
            reg = regime_al(regimi, t, data_segnale) if usa_regime else None
            blocco = usa_regime and (reg is False)
            seg = {k: (False if blocco else v) for k, v in grezzi.items()}
            n_attivi = sum(seg.values())

            chiusura_ieri = float(comp["close"].iloc[-1])
            trigger = float(comp["high"].iloc[-1])
            atr_ieri = float(atr(comp, 14).iloc[-1])

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
                "ticker": t, **seg,
                "n_attivi": n_attivi, "confluenza": n_attivi >= 2,
                "qualcuno": n_attivi >= 1,
                "regime": reg, "spenti_dal_regime": sum(grezzi.values()) if blocco else 0,
                "data_segnale": str(data_segnale.date()),
                "chiusura_ieri": chiusura_ieri, "trigger": trigger,
                "prezzo_ora": prezzo_ora, "massimo_oggi": massimo_oggi,
                "dist_trigger_atr": dist_trigger_atr,
                "dist_chiusura_atr": dist_chiusura_atr,
                "atr_ieri": atr_ieri,
                "stop_indicativo": (max(prezzo_ora, trigger) - 1.5 * atr_ieri),
                "ora_dato": ora_dato, "stato": stato,
            })
        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__})")

    return {"righe": righe, "falliti": falliti,
            "ora_controllo": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "soglia_estensione_atr": soglia_estensione_atr,
            "soglia_annulla_atr": soglia_annulla_atr,
            "usa_regime": usa_regime}
