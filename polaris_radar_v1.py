"""
============================================================
  POLARIS RADAR v1 — Scanner unificato con bias direzionale
  Porta: 8512
  Avvio: Avvia_Polaris_Radar.bat
         oppure: streamlit run polaris_radar_v1.py --server.port 8512

  Pipeline filtri (in ordine):
    [1] POLARIS bias daily  — per-ticker, 6 componenti, score 0–6
    [2] Regime indice       — close > EMA200 (macro)
    [3] Weekly trend        — close weekly > EMA10
    [4] Algoritmi           — Momentum / Pullback / Compressione (v3)
    [5] Score 0–10 + R:R + semaforo staleness

  POLARIS bias:
    c > EMA20 · c > EMA50 · c > EMA200 · EMA20>50 · RSI>50 · HH-HL
    Score 5–6 → BULLISH 🟢 | Score 3–4 → NEUTRAL ⚪ | Score 0–2 → BEARISH 🔴

  Dipendenze: backtest_engine.py + backtest_engine_v3.py (stessa cartella)
  Watchlist:  titoli_radar.json  (identica a radar_v1 — zero migrazione)
============================================================
"""

import os
import json
import shutil
import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st

import backtest_engine_v3 as eng
from backtest_engine import ema, rsi, atr, adx

# ── compatibilità timezone ──────────────────────────────────
try:
    from zoneinfo import ZoneInfo as _ZI
    _TZ_ITA = _ZI("Europe/Rome")
except Exception:
    _TZ_ITA = None


def _ora_ita(ora_it=None):
    """Ora italiana locale."""
    if ora_it is not None:
        return ora_it
    if _TZ_ITA:
        return _dt.datetime.now(_TZ_ITA)
    return _dt.datetime.now()


if hasattr(eng, "_ora_ita"):
    _ora_ita = eng._ora_ita  # type: ignore[assignment]

# ── controllo dipendenze engine v3 ─────────────────────────
_MISSING = [a for a in ("_solo_complete", "_seg_v3", "MIN_CANDELE") if not hasattr(eng, a)]
_solo_complete = getattr(eng, "_solo_complete", None)
_seg_v3        = getattr(eng, "_seg_v3", None)
MIN_CANDELE    = getattr(eng, "MIN_CANDELE", 250)

# ── configurazione pagina ───────────────────────────────────
st.set_page_config(
    page_title="POLARIS Radar — Scanner Unificato",
    page_icon="🌟",
    layout="wide",
)

TIT_FILE = "titoli_radar.json"
NOMI     = {"momentum": "Momentum", "pullback": "Pullback", "compressione": "Compressione"}
ALGOS    = ("momentum", "pullback", "compressione")

LIQUIDITA_MIN = {
    ".MI":     150_000,   # 150 k€/gg per titoli italiani
    "default": 500_000,   # 500 k€/gg per USA / altri
}

ORDINE_STATO = {
    "VIA LIBERA":     0,
    "STALLO":         1,
    "ATTESA TRIGGER": 2,
    "RIENTRATO":      3,
    "NON INSEGUIRE":  4,
    "ANNULLATO":      5,
    "SCADUTO":        6,
    "DATI N/D":       7,
    "—":              8,
}

COLORI_STATO = {
    "VIA LIBERA":     "background-color: rgba(34,197,94,0.22); font-weight:500",
    "ATTESA TRIGGER": "background-color: rgba(59,130,246,0.12)",
    "STALLO":         "background-color: rgba(249,115,22,0.15)",
    "SCADUTO":        "background-color: rgba(120,120,120,0.12)",
    "NON INSEGUIRE":  "background-color: rgba(250,204,21,0.18)",
    "RIENTRATO":      "background-color: rgba(249,115,22,0.15)",
    "ANNULLATO":      "background-color: rgba(239,68,68,0.18)",
}

POLARIS_ICONS = {"BULLISH": "🟢", "NEUTRAL": "⚪", "BEARISH": "🔴"}


# ══════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════

def jload(path):
    if not os.path.exists(path):
        return [], None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


def jsave(path, data):
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def invalida():
    for k in ("scan_radar", "sf_radar", "polaris_overview"):
        st.session_state.pop(k, None)


def prezzo_fmt(p):
    if p is None:
        return "—"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if abs(p) >= 100:
        return f"{p:.2f}"
    if abs(p) >= 1:
        return f"{p:.3f}".rstrip("0").rstrip(".")
    return f"{p:.4f}".rstrip("0").rstrip(".")


def regime_icona(reg):
    if reg is True:  return "🟢"
    if reg is False: return "🔴"
    return "·"


def weekly_icona(w):
    if w is True:  return "🟢"
    if w is False: return "🔴"
    return "·"


def rr_fmt(rr):
    if rr is None:
        return "—"
    if rr >= 2.0:  return f"🟢 {rr:.1f}"
    if rr >= 1.5:  return f"🟡 {rr:.1f}"
    return f"🔴 {rr:.1f}"


def score_bar(s):
    if s is None:
        return "—"
    filled = int(round(s))
    return "█" * filled + "░" * (10 - filled) + f"  {s:.1f}"


def pol_fmt(bias, score):
    """Formatta bias POLARIS per la tabella."""
    icon = POLARIS_ICONS.get(bias, "·")
    return f"{icon} {score}/6"


# ══════════════════════════════════════════════════════════
# POLARIS BIAS — bias direzionale daily per-ticker
# ══════════════════════════════════════════════════════════

def polaris_bias(df):
    """
    Bias direzionale daily a 6 componenti (ispirato a POLARIS Pine Script).
    Usa i dati daily già scaricati — nessun download aggiuntivo.

    Componenti (1 punto ciascuno):
      1. Close > EMA 20
      2. Close > EMA 50
      3. Close > EMA 200
      4. EMA 20 > EMA 50   (allineamento short/medium)
      5. RSI 14 > 50       (momentum positivo)
      6. Struttura HH-HL   (massimo recente = HH e/o minimi in rialzo)

    Score 5–6 → BULLISH 🟢
    Score 3–4 → NEUTRAL ⚪
    Score 0–2 → BEARISH 🔴

    Restituisce dict: {bias, polaris_score, polaris_note}
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    score = 0
    note  = []

    try:
        e20   = ema(c, 20)
        e50   = ema(c, 50)
        e200  = ema(c, 200)
        r14   = rsi(c, 14)

        lc    = float(c.iloc[-1])
        le20  = float(e20.iloc[-1])
        le50  = float(e50.iloc[-1])
        le200 = float(e200.iloc[-1])
        lr    = float(r14.iloc[-1])

        # 1. Close > EMA 20
        if lc > le20:
            score += 1; note.append("c>EMA20")

        # 2. Close > EMA 50
        if lc > le50:
            score += 1; note.append("c>EMA50")

        # 3. Close > EMA 200
        if lc > le200:
            score += 1; note.append("c>EMA200")

        # 4. EMA 20 > EMA 50 (trend alignment)
        if le20 > le50:
            score += 1; note.append("EMA20>50")

        # 5. RSI > 50
        if lr > 50:
            score += 1; note.append(f"RSI {lr:.0f}")

        # 6. Struttura HH-HL (ultimi 20 barre)
        # HH: l'ultima barra è al 99%+ del massimo delle 20
        # HL: i minimi degli ultimi 5gg > i minimi dei 15gg precedenti
        h20  = h.iloc[-20:]
        l20  = l.iloc[-20:]
        hh   = float(h20.iloc[-1]) >= float(h20.iloc[:-1].max()) * 0.99
        hl   = float(l20.iloc[-5:].min()) > float(l20.iloc[:-5].min())
        if hh or hl:
            score += 1; note.append("HH-HL")

    except Exception:
        pass

    if score >= 5:
        bias = "BULLISH"
    elif score >= 3:
        bias = "NEUTRAL"
    else:
        bias = "BEARISH"

    return {
        "bias":          bias,
        "polaris_score": score,
        "polaris_note":  " · ".join(note) if note else "n/d",
    }


# ══════════════════════════════════════════════════════════
# FUNZIONI DI ARRICCHIMENTO (identiche a radar_v1)
# ══════════════════════════════════════════════════════════

def punteggio_segnale(df, n_attivi=1):
    """
    Score 0–10 sulla qualità del contesto al momento del segnale.

    Componenti:
      ADX > 30 → +2 pt   |  ADX > 20 → +1 pt
      Vol > 1.5x → +2 pt |  Vol > 1.0x → +1 pt
      Candela forte (corpo > 60%, rialzista) → +2 pt | positiva → +1 pt
      RSI 50–70 → +1 pt
      Dist. EMA200 in 0–20% → +1 pt
      Confluenza 2+ algos → +2 pt bonus
    """
    c = df["close"]
    v = df["volume"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    score = 0.0
    note  = []

    # ADX
    try:
        ax = float(adx(df, 14).iloc[-1])
        if ax > 30:
            score += 2.0; note.append(f"ADX {ax:.0f}★")
        elif ax > 20:
            score += 1.0; note.append(f"ADX {ax:.0f}")
    except Exception:
        pass

    # Volume vs media 20gg
    try:
        vm = float(v.rolling(20).mean().iloc[-1])
        if vm > 0:
            vr = float(v.iloc[-1]) / vm
            if vr > 1.5:
                score += 2.0; note.append(f"vol {vr:.1f}x★")
            elif vr > 1.0:
                score += 1.0; note.append(f"vol {vr:.1f}x")
    except Exception:
        pass

    # Qualità candela
    try:
        rng   = float(h.iloc[-1] - l.iloc[-1])
        corpo = abs(float(c.iloc[-1] - o.iloc[-1])) / rng if rng > 0 else 0
        if c.iloc[-1] > o.iloc[-1] and corpo > 0.60:
            score += 2.0; note.append(f"candela forte {corpo:.0%}")
        elif c.iloc[-1] > o.iloc[-1]:
            score += 1.0; note.append("candela positiva")
    except Exception:
        pass

    # RSI zona ottimale
    try:
        r = float(rsi(c, 14).iloc[-1])
        if 50 < r < 70:
            score += 1.0; note.append(f"RSI {r:.0f}")
    except Exception:
        pass

    # Distanza da EMA200
    try:
        e200 = float(ema(c, 200).iloc[-1])
        dist = (float(c.iloc[-1]) / e200 - 1) * 100
        if 0 < dist < 20:
            score += 1.0; note.append(f"+{dist:.1f}% EMA200")
        elif dist < 0:
            note.append("sotto EMA200")
        else:
            note.append(f"esteso {dist:.0f}% EMA200")
    except Exception:
        pass

    # Bonus confluenza
    if n_attivi >= 2:
        score += 2.0; note.append("confluenza")

    return {"score": round(min(score, 10.0), 1), "note": " • ".join(note)}


def calcola_rr(df, prezzo_entrata, atr_val, stop_atr=1.5, finestra_target=60):
    """
    Stop   = prezzo_entrata − stop_atr × ATR
    Target = swing high degli ultimi finestra_target giorni
             Fallback: prezzo_entrata + 3 × ATR
    """
    if atr_val <= 0 or prezzo_entrata <= 0:
        return None, None, None
    stop    = prezzo_entrata - stop_atr * atr_val
    rischio = prezzo_entrata - stop
    if rischio <= 0:
        return None, None, None
    swing_high = float(df["high"].iloc[-finestra_target:-1].max())
    if swing_high <= prezzo_entrata:
        swing_high = prezzo_entrata + 3 * atr_val
    rendimento = swing_high - prezzo_entrata
    rr = rendimento / rischio
    return round(rr, 1), round(stop, 4), round(swing_high, 4)


def check_liquidita(df, ticker, giorni=20):
    """Controlla se il valore medio giornaliero supera la soglia di mercato."""
    soglia = LIQUIDITA_MIN["default"]
    for suf, val in LIQUIDITA_MIN.items():
        if suf != "default" and str(ticker).upper().endswith(suf):
            soglia = val
            break
    try:
        val_medio = float((df["volume"] * df["close"]).rolling(giorni).mean().iloc[-1])
        return val_medio >= soglia, round(val_medio / 1_000, 0)
    except Exception:
        return True, 0.0


def trend_weekly(ticker, periodo="2y"):
    """True = close weekly sopra EMA10 (uptrend settimanale)."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period=periodo, interval="1wk",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        c = df["close"].dropna()
        if len(c) < 12:
            return None
        return bool(c.iloc[-1] > ema(c, 10).iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def carica_weekly_cached(tickers_tuple, periodo="2y"):
    """Cached: scarica i weekly una sola volta per sessione (TTL 1h)."""
    return {t: trend_weekly(t, periodo) for t in tickers_tuple}


def qualita_pullback(df, finestra=5):
    """
    True  = volume recente < 90% del volume precedente (pullback sano).
    False = volume in aumento (possibile distribuzione).
    """
    v = df["volume"]
    try:
        vol_rec  = float(v.iloc[-finestra:].mean())
        vol_prec = float(v.iloc[-finestra - 20: -finestra].mean())
        if vol_prec == 0:
            return None
        return bool(vol_rec / vol_prec < 0.90)
    except Exception:
        return None


def calcola_staleness(df_comp, data_segnale, trigger):
    """
    Quante sessioni sono passate da data_segnale senza che
    il massimo giornaliero abbia superato trigger.
    """
    try:
        d0   = pd.Timestamp(data_segnale)
        dopo = df_comp[df_comp.index > d0]
        if dopo.empty:
            return 0, False
        toccato = bool((dopo["high"] >= trigger).any())
        if toccato:
            idx_hit = int(np.argmax((dopo["high"] >= trigger).values))
            return idx_hit, True
        return len(dopo), False
    except Exception:
        return 0, False


# ══════════════════════════════════════════════════════════
# POLARIS OVERVIEW — bias daily per tutta la watchlist
# ══════════════════════════════════════════════════════════

def scansione_polaris(titoli):
    """
    Calcola il bias POLARIS per tutti i ticker in watchlist.
    Utile come vista rapida mattutina prima della scansione segnali.
    """
    tickers = [t["ticker"] for t in titoli]
    dati    = eng.carica_watchlist(tickers, "2y")
    righe   = []
    falliti = []

    for t_info in titoli:
        t  = t_info["ticker"]
        df = dati.get(t)
        if df is None:
            falliti.append(t)
            continue
        try:
            pol    = polaris_bias(df)
            c      = df["close"]
            prezzo = float(c.iloc[-1])
            e20v   = float(ema(c, 20).iloc[-1])
            e50v   = float(ema(c, 50).iloc[-1])
            e200v  = float(ema(c, 200).iloc[-1])
            rsi14  = float(rsi(c, 14).iloc[-1])
            righe.append({
                "ticker":        t,
                "bias":          pol["bias"],
                "polaris_score": pol["polaris_score"],
                "polaris_note":  pol["polaris_note"],
                "prezzo":        prezzo,
                "vs_EMA20":      round((prezzo / e20v  - 1) * 100, 1),
                "vs_EMA50":      round((prezzo / e50v  - 1) * 100, 1),
                "vs_EMA200":     round((prezzo / e200v - 1) * 100, 1),
                "rsi":           round(rsi14, 1),
            })
        except Exception as ex:
            falliti.append(f"{t} ({ex})")

    righe.sort(key=lambda r: -r["polaris_score"])
    return {"righe": righe, "falliti": falliti}


# ══════════════════════════════════════════════════════════
# SCANSIONE PRINCIPALE — estesa con POLARIS
# ══════════════════════════════════════════════════════════

def scansione_radar(
    titoli,
    usa_regime=True,
    usa_weekly=False,
    usa_polaris=True,
    filtra_liquidita=True,
    tieni_oggi=False,
    min_rr=0.0,
    weekly_map=None,
):
    """
    Scan completo con layer POLARIS + arricchimento v1.
    POLARIS bias BULLISH aggiunge +1 pt allo score (cap 10).
    """
    ora_it  = _ora_ita()
    tickers = [t["ticker"] for t in titoli]
    dati    = eng.carica_watchlist(tickers, "2y")
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t not in dati]

    regimi = {}
    if usa_regime:
        regimi = eng.carica_regimi(
            tickers, "2y", tieni_oggi=tieni_oggi, ora_it=ora_it
        )

    righe = []
    for t_info in titoli:
        t  = t_info["ticker"]
        df = dati.get(t)
        if df is None:
            continue
        try:
            comp = _solo_complete(
                df, ticker=t, tieni_oggi=tieni_oggi, ora_it=ora_it
            )
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico corto: {len(comp)} candele)")
                continue

            # ── [1] POLARIS bias ──────────────────────────────
            pol      = polaris_bias(comp)
            pol_bias = pol["bias"]
            pol_sc   = pol["polaris_score"]
            pol_note = pol["polaris_note"]

            if usa_polaris and pol_bias != "BULLISH":
                falliti.append(f"{t} (POLARIS {pol_bias} {pol_sc}/6)")
                continue

            # ── [2] Segnali algoritmi ─────────────────────────
            grezzi = {}
            for algo in ALGOS:
                try:
                    grezzi[algo] = bool(_seg_v3(algo, comp).iloc[-1])
                except Exception:
                    grezzi[algo] = False

            # ── [3] Regime indice ─────────────────────────────
            d_sig  = pd.Timestamp(comp.index[-1])
            reg    = eng.regime_al(regimi, t, d_sig) if usa_regime else None
            blocco = usa_regime and (reg is False)
            seg    = {k: (False if blocco else v) for k, v in grezzi.items()}
            n_att  = sum(seg.values())

            # ── Dati base ─────────────────────────────────────
            prezzo  = float(comp["close"].iloc[-1])
            atr_val = float(atr(comp, 14).iloc[-1])

            # ── [4] Liquidità ─────────────────────────────────
            liq_ok, liq_k = check_liquidita(comp, t)
            if filtra_liquidita and not liq_ok:
                falliti.append(f"{t} (liquidità insufficiente: {liq_k:.0f}k€/gg)")
                continue

            # ── [5] Weekly trend ──────────────────────────────
            w = (weekly_map or {}).get(t) if usa_weekly else None

            # ── Score base ────────────────────────────────────
            sq = (
                punteggio_segnale(comp, n_att)
                if n_att > 0
                else {"score": 0.0, "note": "—"}
            )

            # ── Bonus POLARIS (+1 pt se BULLISH forte 5-6/6) ─
            score_finale = sq["score"]
            note_extra   = sq["note"]
            if pol_sc >= 5 and n_att > 0:
                score_finale = min(round(score_finale + 1.0, 1), 10.0)
                note_extra  += " • POLARIS★"

            # ── R:R ───────────────────────────────────────────
            trigger            = float(comp["high"].iloc[-1])
            rr_val, stop_rr, target_rr = calcola_rr(comp, trigger, atr_val)

            # ── Qualità pullback ──────────────────────────────
            pb_sano = qualita_pullback(comp) if seg.get("pullback") else None

            righe.append({
                "ticker":         t,
                "carattere":      t_info.get("carattere", "calmo"),
                "algoritmo_base": t_info.get("algoritmo", "momentum"),
                # POLARIS
                "pol_bias":       pol_bias,
                "pol_score":      pol_sc,
                "pol_note":       pol_note,
                # Segnali
                **seg,
                "n_attivi":       n_att,
                "confluenza":     n_att >= 2,
                "qualcuno":       n_att > 0,
                "spenti_regime":  sum(grezzi.values()) if blocco else 0,
                "regime":         reg,
                "weekly":         w,
                "data":           str(d_sig.date()),
                "prezzo":         prezzo,
                "atr":            atr_val,
                "trigger":        trigger,
                "stop_rr":        stop_rr,
                "target_rr":      target_rr,
                "rr":             rr_val,
                "score":          score_finale,
                "score_note":     note_extra,
                "liq_k":          liq_k,
                "liq_ok":         liq_ok,
                "pb_sano":        pb_sano,
                "stop_suggerito": round(prezzo - 1.5 * atr_val, 4),
            })

        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__}: {ex})")

    con   = sorted([r for r in righe if r["qualcuno"]], key=lambda r: -r["score"])
    senza = [r for r in righe if not r["qualcuno"]]

    if min_rr > 0:
        con = [r for r in con if r["rr"] is not None and r["rr"] >= min_rr]

    return {
        "con_segnale":        con,
        "senza_segnale":      senza,
        "tutti":              con + senza,
        "falliti":            falliti,
        "data_aggiornamento": str(pd.Timestamp.today().date()),
        "usa_regime":         usa_regime,
        "usa_polaris":        usa_polaris,
        "tieni_oggi":         tieni_oggi,
    }


# ══════════════════════════════════════════════════════════
# SEMAFORO CON STALENESS — esteso con POLARIS
# ══════════════════════════════════════════════════════════

def semaforo_radar(
    titoli,
    usa_regime=True,
    usa_polaris=True,
    tieni_oggi=False,
    soglia_est=1.0,
    soglia_ann=0.5,
    weekly_map=None,
):
    """
    Semaforo d'ingresso con staleness + filtro POLARIS.
    STALLO  → 3–5 giorni senza trigger
    SCADUTO → 6+ giorni senza trigger
    """
    ora_it  = _ora_ita()
    tickers = [t["ticker"] for t in titoli]
    dati    = eng.carica_watchlist(tickers, "2y")
    falliti = [f"{t} (dati non scaricati)" for t in tickers if t not in dati]

    regimi = {}
    if usa_regime:
        regimi = eng.carica_regimi(
            tickers, "2y", tieni_oggi=tieni_oggi, ora_it=ora_it
        )

    righe = []
    for t_info in titoli:
        t  = t_info["ticker"]
        df = dati.get(t)
        if df is None:
            continue
        try:
            comp = _solo_complete(
                df, ticker=t, tieni_oggi=tieni_oggi, ora_it=ora_it
            )
            if len(comp) < MIN_CANDELE:
                falliti.append(f"{t} (storico corto)")
                continue

            # ── POLARIS ───────────────────────────────────────
            pol      = polaris_bias(comp)
            pol_bias = pol["bias"]
            pol_sc   = pol["polaris_score"]
            if usa_polaris and pol_bias != "BULLISH":
                falliti.append(f"{t} (POLARIS {pol_bias})")
                continue

            # ── Segnali ───────────────────────────────────────
            grezzi = {}
            for algo in ALGOS:
                try:
                    grezzi[algo] = bool(_seg_v3(algo, comp).iloc[-1])
                except Exception:
                    grezzi[algo] = False

            d_sig  = pd.Timestamp(comp.index[-1])
            reg    = eng.regime_al(regimi, t, d_sig) if usa_regime else None
            blocco = usa_regime and (reg is False)
            seg    = {k: (False if blocco else v) for k, v in grezzi.items()}
            n_att  = sum(seg.values())

            # ── Livelli ───────────────────────────────────────
            chiusura_ieri = float(comp["close"].iloc[-1])
            trigger       = float(comp["high"].iloc[-1])
            atr_ieri      = float(atr(comp, 14).iloc[-1])

            # ── Intraday ──────────────────────────────────────
            q = eng.quadro_intraday(t, dopo_il=d_sig) if n_att else None
            prezzo_ora   = q["prezzo"]       if q else chiusura_ieri
            massimo_oggi = q["massimo_oggi"] if q else None
            ora_dato     = q["ora_dato"]     if q else "n/d"

            dist_trigger  = (prezzo_ora - trigger)       / atr_ieri if atr_ieri else 0.0
            dist_chiusura = (prezzo_ora - chiusura_ieri) / atr_ieri if atr_ieri else 0.0

            # ── Stato base ────────────────────────────────────
            if n_att == 0:
                stato = "—"
            elif q is None:
                stato = "DATI N/D"
            elif dist_chiusura < -soglia_ann:
                stato = "ANNULLATO"
            elif massimo_oggi is not None and massimo_oggi <= trigger:
                stato = "ATTESA TRIGGER"
            elif prezzo_ora > trigger + soglia_est * atr_ieri:
                stato = "NON INSEGUIRE"
            elif prezzo_ora < trigger:
                stato = "RIENTRATO"
            else:
                stato = "VIA LIBERA"

            # ── Staleness ─────────────────────────────────────
            giorni_att, _ = calcola_staleness(comp, d_sig, trigger)
            stato_stale   = stato
            if stato == "ATTESA TRIGGER":
                if giorni_att >= 6:
                    stato_stale = "SCADUTO"
                elif giorni_att >= 3:
                    stato_stale = "STALLO"

            # ── Arricchimento ─────────────────────────────────
            rr_val, stop_rr, target_rr = calcola_rr(comp, trigger, atr_ieri)
            sq = (
                punteggio_segnale(comp, n_att)
                if n_att > 0
                else {"score": 0.0, "note": "—"}
            )
            w = (weekly_map or {}).get(t)

            righe.append({
                "ticker":          t,
                "pol_bias":        pol_bias,
                "pol_score":       pol_sc,
                **seg,
                "n_attivi":        n_att,
                "confluenza":      n_att >= 2,
                "qualcuno":        n_att > 0,
                "regime":          reg,
                "weekly":          w,
                "data_segnale":    str(d_sig.date()),
                "chiusura_ieri":   chiusura_ieri,
                "trigger":         trigger,
                "prezzo_ora":      prezzo_ora,
                "massimo_oggi":    massimo_oggi,
                "dist_trigger":    dist_trigger,
                "dist_chiusura":   dist_chiusura,
                "atr_ieri":        atr_ieri,
                "stop_indicativo": round(max(prezzo_ora, trigger) - 1.5 * atr_ieri, 4),
                "rr":              rr_val,
                "stop_rr":         stop_rr,
                "target_rr":       target_rr,
                "score":           sq["score"],
                "score_note":      sq["note"],
                "ora_dato":        ora_dato,
                "stato":           stato,
                "stato_stale":     stato_stale,
                "giorni_att":      giorni_att,
            })

        except Exception as ex:
            falliti.append(f"{t} (errore: {type(ex).__name__}: {ex})")

    attivi = [r for r in righe if r["qualcuno"]]
    attivi.sort(key=lambda r: (ORDINE_STATO.get(r["stato_stale"], 9), -r["score"]))

    return {
        "righe":         attivi,
        "falliti":       falliti,
        "ora_controllo": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }


# ══════════════════════════════════════════════════════════
# UI ── HEADER
# ══════════════════════════════════════════════════════════

st.title("🌟 POLARIS Radar — Scanner Unificato")
st.caption(
    "POLARIS bias daily (6 componenti) → Regime indice → Weekly trend → "
    "Momentum · Pullback · Compressione — Score 0–10 · R:R · staleness semaforo. "
    "Prezzi Yahoo Finance (borsa primaria). Materiale didattico, non consiglio finanziario."
)

if _MISSING:
    st.error(
        f"**backtest_engine_v3.py** troppo vecchio: "
        f"mancano `{'`, `'.join(_MISSING)}`. Aggiorna il file e riavvia."
    )
    st.stop()

titoli, err_wl = jload(TIT_FILE)
if err_wl:
    st.error(f"File {TIT_FILE} non leggibile ({err_wl}). Controlla o ripristina il .bak.")
    st.stop()
if titoli is None:
    titoli = []

for t in titoli:
    t.setdefault("algoritmo",  "momentum")
    t.setdefault("volatilita", 0.0)
    t.setdefault("carattere",  "calmo")

ora_ita = _ora_ita()
st.caption(
    f"Ora italiana: {ora_ita.strftime('%H:%M')} — "
    "Candela odierna usata solo dopo la chiusura della borsa primaria "
    "(~17:30 .MI · ~22:00 USA)."
)


# ══════════════════════════════════════════════════════════
# UI ── WATCHLIST
# ══════════════════════════════════════════════════════════

st.subheader("Watchlist")

cc = st.columns([3, 1])
nuovo = cc[0].text_input(
    "Aggiungi titoli (virgola o spazio)",
    placeholder="es. NVDA, ENI.MI, UCG.MI",
    key="rdr_new",
)

if cc[1].button("Aggiungi e classifica", use_container_width=True, key="rdr_add"):
    candidati_raw = nuovo.strip()
    if candidati_raw:
        candidati = [
            x.strip().upper()
            for x in candidati_raw.replace(";", ",").replace(" ", ",").split(",")
            if x.strip()
        ]
        aggiunti, doppi, errori = [], [], []
        for tk in dict.fromkeys(candidati):
            if any(t["ticker"] == tk for t in titoli):
                doppi.append(tk); continue
            try:
                with st.spinner(f"Classifico {tk}…"):
                    info = eng.classifica_titolo(tk)
                titoli.append(info); aggiunti.append(info)
            except Exception as ex:
                errori.append(f"{tk} ({ex})")
        if aggiunti:
            jsave(TIT_FILE, titoli)
            invalida()
            st.success("Aggiunti: " + ", ".join(
                f"{i['ticker']} ({i['carattere']}, vol {i['volatilita']}%)"
                for i in aggiunti
            ))
        if doppi:  st.warning("Già presenti: " + ", ".join(doppi))
        if errori: st.error("Non aggiunti: " + " · ".join(errori))
        if aggiunti: st.rerun()

if not titoli:
    st.info("Watchlist vuota. Aggiungi i titoli qui sopra.")
    st.stop()

calmi   = [t for t in titoli if t.get("algoritmo") == "momentum"]
nervosi = [t for t in titoli if t.get("algoritmo") == "pullback"]
st.caption(
    f"{len(titoli)} titoli · calmi (Momentum): {len(calmi)} · "
    f"nervosi (Pullback/Compressione): {len(nervosi)} · "
    f"soglia nervoso: vol ≥ {eng.SOGLIA_NERVOSO:.0f}%"
)

with st.expander("Vedi / gestisci watchlist"):
    df_wl = pd.DataFrame([{
        "Titolo":    t["ticker"],
        "Carattere": "nervoso" if t.get("algoritmo") == "pullback" else "calmo",
        "Vol. %":    str(t.get("volatilita", 0.0)),
        "Algo base": NOMI.get(t.get("algoritmo", "momentum"), "?"),
        "Benchmark": "FTSE MIB" if t["ticker"].upper().endswith(".MI") else "S&P 500",
    } for t in sorted(titoli, key=lambda x: x.get("volatilita", 0))])
    st.dataframe(df_wl, use_container_width=True, hide_index=True)

    elimina = st.multiselect(
        "Elimina dalla watchlist", [t["ticker"] for t in titoli], key="rdr_del"
    )
    if elimina and st.button("Elimina selezionati", key="rdr_delbtn"):
        titoli = [t for t in titoli if t["ticker"] not in elimina]
        jsave(TIT_FILE, titoli)
        invalida()
        st.rerun()

    if os.path.exists("titoli_segnali.json"):
        if st.button("Importa da titoli_segnali.json", key="rdr_import"):
            try:
                esistenti = {t["ticker"] for t in titoli}
                with open("titoli_segnali.json", encoding="utf-8") as f:
                    src = json.load(f)
                nuovi = [x for x in src if x.get("ticker") not in esistenti]
                titoli.extend(nuovi)
                jsave(TIT_FILE, titoli)
                invalida()
                st.success(f"Importati {len(nuovi)} titoli.")
                st.rerun()
            except Exception as ex:
                st.error(f"Errore import: {ex}")

    st.download_button(
        "Esporta watchlist (JSON)",
        data=json.dumps(titoli, ensure_ascii=False, indent=2),
        file_name="titoli_radar.json",
        mime="application/json",
        use_container_width=True,
        key="rdr_exp",
    )

st.divider()


# ══════════════════════════════════════════════════════════
# UI ── OPZIONI SCANSIONE
# ══════════════════════════════════════════════════════════

st.subheader("Opzioni scansione")

oc = st.columns(5)
usa_polaris = oc[0].checkbox(
    "🌟 Filtro POLARIS", value=True, key="opt_pol",
    help="Mostra solo ticker con bias daily BULLISH (score 5–6/6). "
         "Disattiva per vedere anche NEUTRAL e BEARISH.",
)
usa_regime  = oc[1].checkbox(
    "Filtro regime (indice EMA200)", value=True, key="opt_reg",
)
usa_weekly  = oc[2].checkbox(
    "Trend settimanale", value=False, key="opt_week",
    help="Close weekly > EMA10. Rallenta la prima scansione (cache 1h).",
)
filtra_liq  = oc[3].checkbox(
    "Filtro liquidità minima", value=True, key="opt_liq",
    help=".MI ≥ 150 k€/gg · altri ≥ 500 k€/gg (media 20 sessioni).",
)
tieni_oggi  = oc[4].checkbox(
    "Forza candela di oggi", value=False, key="opt_oggi",
    help="Override: usa la candela intraday anche se mercato aperto.",
)

oc2 = st.columns([1, 3])
min_rr_scan  = oc2[0].number_input(
    "R:R minimo (0 = mostra tutti)",
    min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="opt_rr",
)
mostra_senza = oc2[1].checkbox(
    "Mostra anche i titoli senza segnale", value=False, key="opt_all",
)

st.divider()


# ══════════════════════════════════════════════════════════
# UI ── VISTA POLARIS (bias daily tutta la watchlist)
# ══════════════════════════════════════════════════════════

st.subheader("🌟 Vista POLARIS — Bias direzionale watchlist")
st.caption(
    "Bias daily per ogni ticker, indipendente dai segnali algoritmo. "
    "Usa questa sezione la mattina per filtrare l'universo prima di cercare segnali. "
    "6 componenti: c>EMA20 · c>EMA50 · c>EMA200 · EMA20>50 · RSI>50 · HH-HL. "
    "Score 5–6 = BULLISH 🟢 · 3–4 = NEUTRAL ⚪ · 0–2 = BEARISH 🔴"
)

pol_c1, pol_c2 = st.columns([1, 3])
btn_polaris   = pol_c1.button("🌟  Aggiorna POLARIS", type="primary", use_container_width=True, key="rdr_pol")
solo_bullish  = pol_c2.checkbox("Mostra solo BULLISH", value=False, key="pol_bull")

if btn_polaris:
    with st.spinner(f"Calcolo bias POLARIS per {len(titoli)} titoli…"):
        try:
            st.session_state["polaris_overview"] = scansione_polaris(titoli)
        except Exception as ex:
            st.error(f"Errore POLARIS: {ex}")

pol_ov = st.session_state.get("polaris_overview")
if pol_ov:
    righe_pol = pol_ov["righe"]
    if solo_bullish:
        righe_pol = [r for r in righe_pol if r["bias"] == "BULLISH"]

    n_bull = sum(1 for r in pol_ov["righe"] if r["bias"] == "BULLISH")
    n_neut = sum(1 for r in pol_ov["righe"] if r["bias"] == "NEUTRAL")
    n_bear = sum(1 for r in pol_ov["righe"] if r["bias"] == "BEARISH")
    st.success(
        f"Watchlist: **{len(pol_ov['righe'])}** titoli · "
        f"🟢 BULLISH: **{n_bull}** · ⚪ NEUTRAL: {n_neut} · 🔴 BEARISH: {n_bear}"
    )

    pol_rows = []
    for r in righe_pol:
        pol_rows.append({
            "Titolo":      r["ticker"],
            "POLARIS":     f"{POLARIS_ICONS.get(r['bias'],'·')} {r['bias']}",
            "Score":       f"{r['polaris_score']}/6",
            "Prezzo":      prezzo_fmt(r["prezzo"]),
            "vs EMA20":    f"{r['vs_EMA20']:+.1f}%",
            "vs EMA50":    f"{r['vs_EMA50']:+.1f}%",
            "vs EMA200":   f"{r['vs_EMA200']:+.1f}%",
            "RSI":         f"{r['rsi']:.0f}",
            "Componenti":  r["polaris_note"],
        })

    def stile_polaris(row):
        s = row["POLARIS"]
        if "BULLISH" in s:
            return ["background-color: rgba(34,197,94,0.18)"] * len(row)
        if "BEARISH" in s:
            return ["background-color: rgba(239,68,68,0.10)"] * len(row)
        return [""] * len(row)

    tab_pol = pd.DataFrame(pol_rows)
    try:
        st.dataframe(
            tab_pol.style.apply(stile_polaris, axis=1),
            use_container_width=True, hide_index=True,
        )
    except Exception:
        st.dataframe(tab_pol, use_container_width=True, hide_index=True)

    if pol_ov["falliti"]:
        with st.expander(f"Esclusi / errori ({len(pol_ov['falliti'])})"):
            for f in pol_ov["falliti"]:
                st.caption(f"• {f}")
else:
    st.info("Premi «Aggiorna POLARIS» per il bias direzionale della watchlist.")

st.divider()


# ══════════════════════════════════════════════════════════
# UI ── SEGNALI CON SCORE E R:R
# ══════════════════════════════════════════════════════════

st.subheader("Segnali con score e R:R")
st.caption(
    "Ordinato per Score decrescente (0–10). "
    "POLARIS: 🟢/⚪/🔴 con score /6. "
    "M = Momentum · P = Pullback · C = Compressione · "
    "Pb. = volume pullback in calo · R:R 🟢≥2 · 🟡1.5–2 · 🔴<1.5 · 🔗 = confluenza 2+."
)

btn_scan = st.button(
    "🔍  Cerca segnali", type="primary", use_container_width=False, key="rdr_scan"
)

if btn_scan:
    weekly_map = {}
    if usa_weekly:
        with st.spinner("Scarico dati settimanali (cache 1h)…"):
            weekly_map = carica_weekly_cached(
                tuple(t["ticker"] for t in titoli), "2y"
            )
        st.session_state["_weekly_map"] = weekly_map

    with st.spinner(f"Scansione di {len(titoli)} titoli…"):
        try:
            st.session_state["scan_radar"] = scansione_radar(
                titoli,
                usa_regime=usa_regime,
                usa_weekly=usa_weekly,
                usa_polaris=usa_polaris,
                filtra_liquidita=filtra_liq,
                tieni_oggi=tieni_oggi,
                min_rr=min_rr_scan,
                weekly_map=st.session_state.get("_weekly_map", {}),
            )
        except Exception as ex:
            st.error(f"Errore scansione: {ex}")

res = st.session_state.get("scan_radar")
if res:
    con   = res["con_segnale"]
    tutti = res["tutti"] if mostra_senza else con
    spenti = sum(r.get("spenti_regime", 0) for r in res["tutti"])

    riga_info = (
        f"Scansione del {res['data_aggiornamento']} — "
        f"POLARIS: {'attivo' if res['usa_polaris'] else 'spento'} · "
        f"regime: {'attivo' if res['usa_regime'] else 'spento'}"
    )
    if spenti:
        riga_info += f" · spenti dal regime: {spenti}"
    if min_rr_scan > 0:
        riga_info += f" · R:R min: {min_rr_scan:.1f}"
    st.caption(riga_info)

    if not con:
        msg = "Nessun segnale oggi."
        if spenti:
            msg += f" (Il regime ne ha bloccati {spenti}.)"
        st.info(msg)
    else:
        n_conf = sum(1 for r in con if r["confluenza"])
        msg = f"**{len(con)}** titoli con segnale"
        if n_conf:
            msg += f" · confluenza 2+: **{n_conf}**"
        st.success(msg)

    def spunta(b): return "✅" if b else "—"
    def pb_ico(pb): return "🟢" if pb is True else ("🔴" if pb is False else "·")

    tab_rows = []
    for r in tutti:
        tab_rows.append({
            "":        "🔗" if r["confluenza"] else "",
            "Titolo":  r["ticker"],
            "POLARIS": pol_fmt(r.get("pol_bias", ""), r.get("pol_score", 0)),
            "Reg.":    regime_icona(r["regime"]),
            "Weekly":  weekly_icona(r.get("weekly")) if usa_weekly else "·",
            "M":       spunta(r["momentum"]),
            "P":       spunta(r["pullback"]),
            "C":       spunta(r["compressione"]),
            "Pb.":     pb_ico(r.get("pb_sano")),
            "Data":    r["data"],
            "Prezzo":  prezzo_fmt(r["prezzo"]),
            "Trigger": prezzo_fmt(r["trigger"]),
            "Stop":    prezzo_fmt(r["stop_rr"]),
            "Target":  prezzo_fmt(r["target_rr"]),
            "R:R":     rr_fmt(r["rr"]),
            "Score":   score_bar(r["score"]) if r["qualcuno"] else "—",
        })

    tab = pd.DataFrame(tab_rows)

    def stile_scan(row):
        base = (
            "background-color: rgba(34,197,94,0.15); font-weight:500"
            if row[""] == "🔗"
            else ""
        )
        return [base] * len(row)

    try:
        st.dataframe(
            tab.style.apply(stile_scan, axis=1),
            use_container_width=True, hide_index=True,
        )
    except Exception:
        st.dataframe(tab, use_container_width=True, hide_index=True)

    with st.expander("Dettaglio score per un titolo"):
        ticker_sel = st.selectbox(
            "Titolo", options=[r["ticker"] for r in con], key="rdr_det"
        )
        if ticker_sel:
            r_sel = next((r for r in con if r["ticker"] == ticker_sel), None)
            if r_sel:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Score",   f"{r_sel['score']:.1f} / 10")
                c2.metric("POLARIS", f"{r_sel.get('pol_score', '—')}/6")
                c3.metric("R:R",     f"{r_sel['rr']:.1f}" if r_sel["rr"] else "—")
                c4.metric("Stop",    prezzo_fmt(r_sel["stop_rr"]))
                c5.metric("Target",  prezzo_fmt(r_sel["target_rr"]))
                st.caption(f"Note score: {r_sel['score_note']}")
                st.caption(f"POLARIS componenti: {r_sel.get('pol_note', '—')}")

    if res["falliti"]:
        with st.expander(f"Esclusi / errori ({len(res['falliti'])})"):
            for f in res["falliti"]:
                st.caption(f"• {f}")
else:
    st.info("Premi «Cerca segnali» per avviare la scansione.")

st.divider()


# ══════════════════════════════════════════════════════════
# UI ── SEMAFORO CON STALENESS
# ══════════════════════════════════════════════════════════

st.subheader("Semaforo d'ingresso")
st.caption(
    "Trigger = massimo dell'ultima candela completa. "
    "Stato: VIA LIBERA → STALLO (3–5 gg) → SCADUTO (6+ gg) → … "
    "Prezzi intraday Yahoo (~15 min di ritardo)."
)

sf_col1, sf_col2, sf_col3 = st.columns([1, 1, 2])
btn_sf       = sf_col1.button(
    "🚦  Aggiorna semaforo", type="primary", use_container_width=True, key="rdr_sf"
)
solo_conf_sf = sf_col2.checkbox("Solo confluenze (2+)", value=False, key="rdr_sfconf")

with sf_col3.expander("Soglie ATR"):
    s_est = st.slider("NON INSEGUIRE oltre (×ATR)", 0.3, 2.0, 1.0, 0.1, key="sf_est")
    s_ann = st.slider("ANNULLATO sotto (×ATR)",     0.2, 1.5, 0.5, 0.1, key="sf_ann")

if btn_sf:
    w_map = st.session_state.get("_weekly_map", {})
    with st.spinner("Semaforo in aggiornamento…"):
        try:
            st.session_state["sf_radar"] = semaforo_radar(
                titoli,
                usa_regime=usa_regime,
                usa_polaris=usa_polaris,
                tieni_oggi=tieni_oggi,
                soglia_est=s_est,
                soglia_ann=s_ann,
                weekly_map=w_map if usa_weekly else None,
            )
        except Exception as ex:
            st.error(f"Errore semaforo: {ex}")

sf_res = st.session_state.get("sf_radar")
if sf_res:
    righe_sf = sf_res["righe"]
    if solo_conf_sf:
        righe_sf = [r for r in righe_sf if r["confluenza"]]

    st.caption(f"Aggiornato alle {sf_res['ora_controllo']} — Yahoo ~15 min ritardo.")

    if not righe_sf:
        msg = "Nessun segnale da sorvegliare"
        if solo_conf_sf:
            msg += " (con confluenza)"
        st.info(msg + ".")
    else:
        n_via   = sum(1 for r in righe_sf if r["stato_stale"] == "VIA LIBERA")
        n_att   = sum(1 for r in righe_sf if r["stato_stale"] == "ATTESA TRIGGER")
        n_stall = sum(1 for r in righe_sf if r["stato_stale"] == "STALLO")
        n_scad  = sum(1 for r in righe_sf if r["stato_stale"] == "SCADUTO")
        st.success(
            f"Sorvegliati: **{len(righe_sf)}** · "
            f"VIA LIBERA: {n_via} · ATTESA: {n_att} · "
            f"STALLO: {n_stall} · SCADUTO: {n_scad}"
        )

    sf_rows = []
    for r in righe_sf:
        gg_txt = (
            str(r["giorni_att"])
            if r["stato_stale"] in ("ATTESA TRIGGER", "STALLO", "SCADUTO")
            else "—"
        )
        sf_rows.append({
            "":           "🔗" if r["confluenza"] else "",
            "Titolo":     r["ticker"],
            "POLARIS":    pol_fmt(r.get("pol_bias", ""), r.get("pol_score", 0)),
            "Reg.":       regime_icona(r["regime"]),
            "Weekly":     weekly_icona(r.get("weekly")) if usa_weekly else "·",
            "M":          "✅" if r["momentum"]    else "—",
            "P":          "✅" if r["pullback"]     else "—",
            "C":          "✅" if r["compressione"] else "—",
            "Data segn.": r["data_segnale"],
            "Trigger":    prezzo_fmt(r["trigger"]),
            "Ora":        prezzo_fmt(r["prezzo_ora"]),
            "Max oggi":   prezzo_fmt(r["massimo_oggi"]) if r.get("massimo_oggi") else "—",
            "Δ (ATR)":    f"{r['dist_trigger']:+.2f}",
            "Stop":       prezzo_fmt(r["stop_indicativo"]),
            "R:R":        rr_fmt(r["rr"]),
            "Score":      f"{r['score']:.1f}",
            "Gg att.":    gg_txt,
            "Stato":      r["stato_stale"],
        })

    tab_sf = pd.DataFrame(sf_rows)

    def colora_sf(row):
        c = COLORI_STATO.get(row["Stato"], "")
        return [c] * len(row)

    try:
        st.dataframe(
            tab_sf.style.apply(colora_sf, axis=1),
            use_container_width=True, hide_index=True,
        )
    except Exception:
        st.dataframe(tab_sf, use_container_width=True, hide_index=True)

    st.caption(
        "🟩 VIA LIBERA = trigger superato, prezzo in zona · "
        "🟦 ATTESA TRIGGER = max di ieri non rotto · "
        "🟧 STALLO = 3–5 gg senza trigger · "
        "⬜ SCADUTO = 6+ gg (momentum perso, da archiviare) · "
        "🟨 NON INSEGUIRE = esteso oltre soglia ATR · "
        "🟥 ANNULLATO = sceso troppo dalla chiusura. "
        "«Gg att.» = sessioni dal segnale senza trigger."
    )

    if sf_res["falliti"]:
        with st.expander(f"Esclusi / errori ({len(sf_res['falliti'])})"):
            for f in sf_res["falliti"]:
                st.caption(f"• {f}")
else:
    st.info("Premi «Aggiorna semaforo» a mercato aperto per il controllo intraday.")

st.divider()
st.caption(
    "Watchlist in titoli_radar.json · backup .bak automatico. "
    "Prezzi Yahoo Finance (borsa primaria, non Gettex). "
    "Lo strumento segnala, la decisione è tua."
)
