"""
============================================================
 ENTRY QUALITY ENGINE v5 - timing entrate + regola volatilita'
============================================================
Misura il solo timing d'ingresso (cosa fa il prezzo DOPO il segnale)
per tre entrate, su una watchlist, e testa l'ipotesi:
  titoli CALMI -> meglio il Momentum ; titoli NERVOSI -> meglio il Pullback.
============================================================
"""
import numpy as np
import pandas as pd

MIN_SEGNALI = 15


def carica_dati(ticker, periodo="max"):
    import yfinance as yf
    df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato per '{ticker}'. Borsa italiana: aggiungi .MI (es. ENI.MI).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ---- indicatori ----
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    su = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    giu = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + su/giu.replace(0, np.nan))

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd_hist(close):
    return (ema(close, 12) - ema(close, 26)) - ema(ema(close, 12) - ema(close, 26), 9)

def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up, down = h.diff(), -l.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/a
    mdi = 100*pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/a
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def barre_da_max(highs, finestra):
    return highs.rolling(finestra).apply(lambda x: (len(x)-1) - np.argmax(x), raw=True)


# ---- i tre segnali ----
def seg_pullback(df):
    c, o, h = df["close"], df["open"], df["high"]
    e21, e50, e200 = ema(c, 21), ema(c, 50), ema(c, 200)
    trend = (c > e200) & (e21 > e50) & (e50 > e200)
    maxrec = h.rolling(20).max()
    pull = ((maxrec - c)/maxrec*100 >= 1.5) & (barre_da_max(h, 20) >= 4)
    r, hist, ax = rsi(c, 14), macd_hist(c), adx(df, 14)
    score = (((r > 40) & (r < 75)).astype(int) + ((hist < 0) & (hist > hist.shift())).astype(int) +
             (c > o).astype(int) + ((c > e21*0.98) & (c < e21*1.03)).astype(int) + (ax > 18).astype(int))
    a = atr(df, 14); veto = a > a.rolling(50).mean()*2.5
    return (trend & pull & (score >= 3) & (~veto)).fillna(False)

def seg_compressione(df):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    e50 = ema(c, 50); sopra = c >= e50
    r = rsi(c, 14)
    c_vol = v.rolling(3).mean() < v.shift(3).rolling(3).mean()*0.85
    c_range = (h - l)/c*100 < 0.8
    cnt = ((r >= 30) & (r <= 60)).astype(int) + c_vol.astype(int) + c_range.astype(int) + sopra.astype(int)
    setup2 = (cnt >= 3) & (cnt >= 3).shift(1)
    corpo = (c - o).abs()/(h - l).replace(0, np.nan)*100
    breakout = (c > o) & (corpo >= 60) & (v >= v.rolling(20).mean()*1.5)
    return ((setup2 | breakout) & sopra).fillna(False)

def seg_momentum(df):
    c, v = df["close"], df["volume"]
    return ((ema(c, 20) > ema(c, 50)) & (rsi(c, 14) > 50) & (adx(df, 14) > 20) &
            (v > v.rolling(20).mean()*1.2)).fillna(False)

ENTRATE = {"Pullback (SwingNotte)": seg_pullback,
           "Compressione (Rally)": seg_compressione,
           "Momentum (mio)": seg_momentum}


# ---- radiografia dopo-segnale ----
def analizza_entrata(df, segnale, orizzonti=(1, 3, 5), target_pct=3.0):
    entry = df["open"].shift(-1)
    H = max(orizzonti)
    sig = segnale & entry.notna()
    n = int(sig.sum())
    out = {"n_segnali": n}
    if n == 0:
        return out
    for h in orizzonti:
        fwd = (df["close"].shift(-h)/entry - 1)*100
        out[f"ret_{h}"] = float(fwd[sig].mean())
        out[f"base_{h}"] = float(fwd.mean())
        out[f"edge_{h}"] = out[f"ret_{h}"] - out[f"base_{h}"]
    fwdH = (df["close"].shift(-H)/entry - 1)*100
    out[f"pos_{H}"] = float((fwdH[sig] > 0).mean()*100)
    fut_high = pd.concat([df["high"].shift(-k) for k in range(1, H+1)], axis=1).max(axis=1)
    fut_low  = pd.concat([df["low"].shift(-k)  for k in range(1, H+1)], axis=1).min(axis=1)
    out["hit_target"] = float((fut_high >= entry*(1+target_pct/100))[sig].mean()*100)
    out["mae"] = float(((fut_low/entry - 1)*100)[sig].mean())
    return out

def confronta_tutte(df, orizzonti=(1, 3, 5), target_pct=3.0):
    return {nome: analizza_entrata(df, fn(df), orizzonti, target_pct) for nome, fn in ENTRATE.items()}


# ---- volatilita' di un titolo (ATR medio in %) ----
def volatilita_pct(df):
    return float((atr(df, 14)/df["close"]*100).dropna().mean())


# ==========================================================
# WATCHLIST
# ==========================================================
def carica_watchlist(tickers, periodo="5y"):
    import yfinance as yf
    out = {}
    tickers = [t for t in tickers if t]
    if not tickers:
        return out
    data = yf.download(tickers, period=periodo, auto_adjust=True, progress=False, group_by="ticker")
    for t in tickers:
        try:
            sub = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub = sub[["open", "high", "low", "close", "volume"]].dropna()
            if len(sub) > 250:
                out[t] = sub
        except Exception:
            continue
    return out


def analizza_dati(dati, orizzonti=(1, 3, 5), target_pct=3.0):
    """{ticker: df} -> {ticker: {entrata: metriche}}"""
    per_titolo = {}
    for t, df in dati.items():
        try:
            per_titolo[t] = confronta_tutte(df, orizzonti, target_pct)
        except Exception:
            continue
    return per_titolo


def riepiloga(per_titolo):
    nomi = list(ENTRATE.keys())
    riep = {n: {"vince_su": 0} for n in nomi}
    for nome in nomi:
        edges, hits, seg_tot, pos, validi = [], [], 0, 0, 0
        for ent in per_titolo.values():
            e = ent.get(nome, {}); ns = e.get("n_segnali", 0); seg_tot += ns
            if ns >= MIN_SEGNALI and e.get("edge_5") is not None:
                validi += 1; edges.append(e["edge_5"]); hits.append(e.get("hit_target", 0.0))
                if e["edge_5"] > 0: pos += 1
        riep[nome].update({"edge_medio": float(np.mean(edges)) if edges else float("nan"),
                           "hit_medio": float(np.mean(hits)) if hits else float("nan"),
                           "segnali_tot": seg_tot, "positivo_su": pos, "titoli_validi": validi})
    for ent in per_titolo.values():
        cand = {n: ent[n]["edge_5"] for n in nomi
                if ent.get(n, {}).get("n_segnali", 0) >= MIN_SEGNALI and ent[n].get("edge_5") is not None}
        if cand:
            riep[max(cand, key=cand.get)]["vince_su"] += 1
    return riep


# ==========================================================
# REGOLA VOLATILITA' -> entrata
# ==========================================================
def regola_volatilita(dati, per_titolo):
    P, M = "Pullback (SwingNotte)", "Momentum (mio)"
    righe = []
    for t, df in dati.items():
        ent = per_titolo.get(t, {})
        ep, em = ent.get(P, {}), ent.get(M, {})
        if ep.get("n_segnali", 0) < MIN_SEGNALI or em.get("n_segnali", 0) < MIN_SEGNALI:
            continue
        if ep.get("edge_5") is None or em.get("edge_5") is None:
            continue
        righe.append({"ticker": t, "vol": round(volatilita_pct(df), 2),
                      "edge_pull": round(ep["edge_5"], 2), "edge_mom": round(em["edge_5"], 2)})
    if len(righe) < 4:
        return {"ok": False, "n": len(righe), "righe": righe}

    d = pd.DataFrame(righe)
    d["diff_mom_pull"] = (d["edge_mom"] - d["edge_pull"]).round(2)
    d["vincitore"] = np.where(d["edge_mom"] >= d["edge_pull"], "Momentum", "Pullback")
    d = d.sort_values("vol").reset_index(drop=True)
    med = d["vol"].median()
    calmi, nervosi = d[d["vol"] <= med], d[d["vol"] > med]

    def grp(g):
        return {"n": int(len(g)), "vol_media": float(g["vol"].mean()),
                "edge_mom": float(g["edge_mom"].mean()), "edge_pull": float(g["edge_pull"].mean()),
                "mom_vince": int((g["vincitore"] == "Momentum").sum()),
                "pull_vince": int((g["vincitore"] == "Pullback").sum())}

    gc, gn = grp(calmi), grp(nervosi)
    corr = float(d["vol"].corr(d["diff_mom_pull"]))
    # ipotesi confermata se: nei calmi vince il Momentum, nei nervosi vince il Pullback
    conferma = (gc["edge_mom"] > gc["edge_pull"]) and (gn["edge_pull"] > gn["edge_mom"])
    return {"ok": True, "n": len(d), "righe": d.to_dict("records"),
            "mediana_vol": float(med), "calmi": gc, "nervosi": gn,
            "correlazione": corr, "conferma": bool(conferma)}


# ==========================================================
# TEST DELLA REGOLA: volatilita' -> entrata giusta
#   ipotesi: titoli tranquilli -> Momentum ; titoli nervosi -> Pullback
#   (regola STRUTTURALE, decisa dalla volatilita', non dai vincitori passati)
# ==========================================================
def volatilita_annua(df):
    """'Nervosismo' del titolo: deviazione dei rendimenti giornalieri, annualizzata, in %."""
    r = np.log(df["close"] / df["close"].shift()).dropna()
    return float(r.std() * np.sqrt(252) * 100)


def testa_regola_volatilita(tickers, periodo="5y", orizzonti=(1, 3, 5), target_pct=3.0, soglia=None):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []
    for t, df in dati.items():
        try:
            ap = analizza_entrata(df, seg_pullback(df), orizzonti, target_pct)
            am = analizza_entrata(df, seg_momentum(df), orizzonti, target_pct)
            righe.append({"ticker": t, "vol": volatilita_annua(df),
                          "edge_pull": ap.get("edge_5"), "n_pull": ap.get("n_segnali", 0),
                          "edge_mom": am.get("edge_5"), "n_mom": am.get("n_segnali", 0)})
        except Exception:
            falliti.append(t)

    # set comune: solo titoli dove ENTRAMBE le entrate hanno abbastanza segnali (confronto equo)
    comuni = [r for r in righe if r["n_pull"] >= MIN_SEGNALI and r["n_mom"] >= MIN_SEGNALI
              and r["edge_pull"] is not None and r["edge_mom"] is not None]
    if soglia is None and comuni:
        soglia = float(np.median([r["vol"] for r in comuni]))   # soglia = mediana (non ottimizzata sul risultato)

    media = lambda k: float(np.mean([r[k] for r in comuni])) if comuni else float("nan")
    edge_sempre_mom = media("edge_mom")
    edge_sempre_pull = media("edge_pull")

    scelte = []
    for r in comuni:
        scelto = "momentum" if (soglia is not None and r["vol"] < soglia) else "pullback"
        scelte.append({**r, "scelto": scelto,
                       "edge_scelto": r["edge_mom"] if scelto == "momentum" else r["edge_pull"]})
    edge_regola = float(np.mean([s["edge_scelto"] for s in scelte])) if scelte else float("nan")

    if len(comuni) >= 4:
        xs = np.array([r["vol"] for r in comuni])
        ys = np.array([r["edge_pull"] - r["edge_mom"] for r in comuni])
        corr = float(np.corrcoef(xs, ys)[0, 1])
    else:
        corr = float("nan")

    return {"righe": sorted(scelte, key=lambda r: r["vol"]), "soglia": soglia,
            "edge_sempre_mom": edge_sempre_mom, "edge_sempre_pull": edge_sempre_pull,
            "edge_regola": edge_regola, "corr": corr, "n_comuni": len(comuni),
            "falliti": falliti, "target_pct": target_pct}


# ==========================================================
# GESTIONE DELL'USCITA con COSTI - test su watchlist
#   entrata fissa (Momentum o Pullback), si confrontano i modi di uscire:
#   - "target"   : vendi al +X% (con stop e uscita a tempo)
#   - "trailing" : lascia correre (paletto ATR che insegue), niente tetto
#   Tutto AL NETTO dei costi (commissioni + spread).
# ==========================================================
def backtest_pnl(df, segnale, cfg, capitale=10_000):
    cap = capitale
    in_pos = False
    entrata = stop = target = qty = trail = 0.0
    giorni = 0
    trade, esiti = [], {}
    r = df.reset_index(); n = len(r)
    a = atr(df, 14).values
    op, hi, lo, cl = r["open"].values, r["high"].values, r["low"].values, r["close"].values
    seg = segnale.values
    costo = cfg.get("costo_pct", 0.2) / 100

    equity = [capitale]

    def chiudi(prezzo, esito):
        nonlocal cap
        lordo = (prezzo - entrata) * qty
        costo_eur = entrata * qty * costo            # frizione su entrata+uscita
        cap += lordo - costo_eur
        trade.append(lordo - costo_eur)
        equity.append(cap)
        esiti[esito] = esiti.get(esito, 0) + 1

    for i in range(n - 1):
        if not in_pos:
            if seg[i] and not np.isnan(a[i]) and a[i] > 0:
                entrata = op[i+1]
                stop = entrata - cfg.get("stop_atr", 1.5) * a[i]
                if stop >= entrata:
                    stop = entrata * (1 - 0.005)
                target = entrata * (1 + cfg.get("target_pct", 4.0)/100)
                risch = entrata - stop
                if risch <= 0:
                    continue
                qty = (cap * cfg.get("rischio_pct", 0.01)) / risch
                trail = stop
                in_pos = True; giorni = 0
        else:
            giorni += 1
            uscita = esito = None
            if cfg["modo"] == "trailing":
                if not np.isnan(a[i]):
                    trail = max(trail, hi[i] - cfg.get("trail_atr", 3.0) * a[i])
                if lo[i] <= trail:
                    uscita, esito = trail, "uscita"
            else:
                if lo[i] <= stop:
                    uscita, esito = stop, "stop"
                elif hi[i] >= target:
                    uscita, esito = target, "target"
                elif giorni >= cfg.get("max_giorni", 5):
                    uscita, esito = cl[i], "tempo"
            if uscita is not None:
                chiudi(uscita, esito); in_pos = False; giorni = 0

    if in_pos:
        chiudi(cl[n-1], "fine")
    eq = pd.Series(equity, dtype=float)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100) if len(eq) > 1 else 0.0
    anni = max(len(df) / 252.0, 0.5)
    return {"trade": trade, "rendimento_pct": (cap/capitale-1)*100, "n_trade": len(trade),
            "esiti": esiti, "max_dd": dd, "anni": anni}


def _bh(df):
    return (df["close"].iloc[-1]/df["close"].iloc[0]-1)*100


def confronta_uscite(tickers, periodo="5y", entry="momentum", target_pct=4.0,
                     max_giorni=5, trail_atr=3.0, stop_atr=1.5, rischio_pct=0.01, costo_pct=0.2):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    seg_fn = seg_momentum if entry == "momentum" else seg_pullback
    base = dict(target_pct=target_pct, max_giorni=max_giorni, trail_atr=trail_atr,
                stop_atr=stop_atr, rischio_pct=rischio_pct, costo_pct=costo_pct)
    stili = {"Target fisso": {**base, "modo": "target"},
             "Lascia correre": {**base, "modo": "trailing"}}

    agg = {s: {"rend": [], "ntr": 0, "vinte": 0, "tot": 0, "esiti": {}} for s in stili}
    bh = []
    for t, df in dati.items():
        try:
            sg = seg_fn(df)
            bh.append(_bh(df))
            for s, cfg in stili.items():
                res = backtest_pnl(df, sg, cfg)
                agg[s]["rend"].append(res["rendimento_pct"])
                agg[s]["ntr"] += res["n_trade"]
                agg[s]["vinte"] += sum(1 for x in res["trade"] if x > 0)
                agg[s]["tot"] += len(res["trade"])
                for k, v in res["esiti"].items():
                    agg[s]["esiti"][k] = agg[s]["esiti"].get(k, 0) + v
        except Exception:
            falliti.append(t)

    riep = {}
    for s, d in agg.items():
        riep[s] = {
            "rend_medio": float(np.mean(d["rend"])) if d["rend"] else float("nan"),
            "n_trade": d["ntr"],
            "win_rate": (d["vinte"]/d["tot"]*100) if d["tot"] else float("nan"),
            "esiti": d["esiti"],
        }
    return {"riepilogo": riep, "bh_medio": float(np.mean(bh)) if bh else float("nan"),
            "n_titoli": len(dati), "entry": entry, "costo_pct": costo_pct, "falliti": falliti}


# ==========================================================
# TEST ONESTO sul TUO obiettivo: mini-vantaggio?
#   ogni gruppo con la SUA entrata + uscita "lascia correre", coi costi.
#   Successo = netto positivo dopo i costi E rischio (drawdown) piccolo.
#   (NON si misura contro il compra-e-tieni: non e' il tuo scopo.)
# ==========================================================
def valuta_gruppo(tickers, periodo, entry, trail_atr=3.0, stop_atr=1.5,
                  rischio_pct=0.01, costo_pct=0.2):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    seg_fn = seg_momentum if entry == "momentum" else seg_pullback
    cfg = {"modo": "trailing", "trail_atr": trail_atr, "stop_atr": stop_atr,
           "rischio_pct": rischio_pct, "costo_pct": costo_pct, "target_pct": 4.0, "max_giorni": 5}

    rend, dd, anni, pos, ntr, vinte, tot = [], [], [], 0, 0, 0, 0
    bh, per_titolo = [], []
    for t, df in dati.items():
        try:
            res = backtest_pnl(df, seg_fn(df), cfg)
            rend.append(res["rendimento_pct"]); dd.append(res["max_dd"]); anni.append(res["anni"])
            if res["rendimento_pct"] > 0:
                pos += 1
            ntr += res["n_trade"]
            vinte += sum(1 for x in res["trade"] if x > 0); tot += res["n_trade"]
            bh.append(_bh(df))
            per_titolo.append({"ticker": t, "rend": res["rendimento_pct"],
                               "dd": res["max_dd"], "n": res["n_trade"]})
        except Exception:
            falliti.append(t)

    n = len(rend)
    if n == 0:
        return {"n_titoli": 0, "falliti": falliti}
    rend_medio = float(np.mean(rend))
    anni_medi = float(np.mean(anni)) if anni else 1.0
    return {
        "n_titoli": n, "entry": entry,
        "rend_medio": rend_medio,
        "rend_annuo": rend_medio / anni_medi,
        "trade_anno": (ntr / n) / anni_medi if n else 0.0,
        "max_dd_medio": float(np.mean(dd)),
        "pos_su": pos, "win_rate": (vinte/tot*100) if tot else float("nan"),
        "n_trade": ntr, "bh_medio": float(np.mean(bh)) if bh else float("nan"),
        "per_titolo": sorted(per_titolo, key=lambda x: x["rend"], reverse=True),
        "falliti": falliti,
    }


def test_due_gruppi(calmi, nervosi, periodo="5y", trail_atr=3.0, stop_atr=1.5,
                    rischio_pct=0.01, costo_pct=0.2):
    return {
        "calmi": valuta_gruppo(calmi, periodo, "momentum", trail_atr, stop_atr, rischio_pct, costo_pct),
        "nervosi": valuta_gruppo(nervosi, periodo, "pullback", trail_atr, stop_atr, rischio_pct, costo_pct),
    }


# ==========================================================
# TEST ONESTO (metro giusto): c'e' un mini-vantaggio?
#   - Calmi  -> Momentum ; Nervosi -> Pullback  (scelti a mano)
#   - gestione: "lascia correre" (la migliore), costi inclusi
#   - giudice: ASPETTATIVA per operazione > 0 (non il compra-e-tieni)
# ==========================================================
def test_onesto(calmi, nervosi, periodo="5y", costo_pct=0.2,
                trail_atr=3.0, stop_atr=1.5, rischio_pct=0.01):
    assegna = {t: "momentum" for t in calmi}
    assegna.update({t: "pullback" for t in nervosi})
    tickers = list(assegna.keys())
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    cfg = dict(modo="trailing", trail_atr=trail_atr, stop_atr=stop_atr,
               rischio_pct=rischio_pct, costo_pct=costo_pct, target_pct=4.0, max_giorni=5)

    per = []
    for t, df in dati.items():
        seg = seg_momentum(df) if assegna[t] == "momentum" else seg_pullback(df)
        res = backtest_pnl(df, seg, cfg)
        tr = res["trade"]; n = len(tr)
        anni = max(len(df) / 252, 0.5)
        rend = res["rendimento_pct"]
        ann = ((1 + rend/100) ** (1/anni) - 1) * 100 if rend > -100 else -100.0
        if n:
            eq = 10000 + np.cumsum(tr)
            peak = np.maximum.accumulate(eq)
            dd = float(((eq - peak) / peak).min() * 100)
        else:
            dd = 0.0
        per.append({"ticker": t, "gruppo": assegna[t], "n": n, "rend": rend, "ann": ann,
                    "exp_pct": (float(np.mean(tr)) / 10000 * 100) if n else float("nan"),
                    "netto": float(sum(tr)), "dd": dd,
                    "vinte": sum(1 for x in tr if x > 0)})

    def agg(rows):
        rows = [r for r in rows if r["n"] > 0]
        if not rows:
            return None
        tot_tr = sum(r["n"] for r in rows)
        tot_net = sum(r["netto"] for r in rows)
        tot_win = sum(r["vinte"] for r in rows)
        return {"n_titoli": len(rows), "trade": tot_tr,
                "exp_pct": tot_net / tot_tr / 10000 * 100 if tot_tr else float("nan"),
                "win_rate": tot_win / tot_tr * 100 if tot_tr else float("nan"),
                "rend_medio": float(np.mean([r["rend"] for r in rows])),
                "ann_medio": float(np.mean([r["ann"] for r in rows])),
                "dd_medio": float(np.mean([r["dd"] for r in rows])),
                "positivi": sum(1 for r in rows if r["netto"] > 0)}

    return {"per": per, "tutti": agg(per),
            "calmi": agg([r for r in per if r["gruppo"] == "momentum"]),
            "nervosi": agg([r for r in per if r["gruppo"] == "pullback"]),
            "falliti": falliti, "costo_pct": costo_pct}


# ==========================================================
# OTTIMIZZATORE ONESTO: tara sulla prima meta', giudica sulla seconda
# ==========================================================
def seg_pullback_opt(df, pb_prof=1.5, punteggio_min=3, pb_barre=4):
    c, o, h = df["close"], df["open"], df["high"]
    e21, e50, e200 = ema(c, 21), ema(c, 50), ema(c, 200)
    trend = (c > e200) & (e21 > e50) & (e50 > e200)
    maxrec = h.rolling(20).max()
    dist = (maxrec - c) / maxrec * 100
    pull = (dist >= pb_prof) & (barre_da_max(h, 20) >= pb_barre)
    r, hist, ax = rsi(c, 14), macd_hist(c), adx(df, 14)
    score = (((r > 40) & (r < 75)).astype(int) + ((hist < 0) & (hist > hist.shift())).astype(int) +
             (c > o).astype(int) + ((c > e21*0.98) & (c < e21*1.03)).astype(int) + (ax > 18).astype(int))
    a = atr(df, 14); veto = a > a.rolling(50).mean() * 2.5
    return (trend & pull & (score >= punteggio_min) & (~veto)).fillna(False)


def seg_momentum_opt(df, adx_min=20, rsi_min=50, vol_mult=1.2):
    c, v = df["close"], df["volume"]
    e20, e50 = ema(c, 20), ema(c, 50)
    return ((e20 > e50) & (rsi(c, 14) > rsi_min) & (adx(df, 14) > adx_min) &
            (v > v.rolling(20).mean() * vol_mult)).fillna(False)


def seg_compress_opt(df, range_max=0.8):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    e50 = ema(c, 50); sopra = c >= e50; r = rsi(c, 14)
    vol3 = v.rolling(3).mean(); volprec3 = v.shift(3).rolling(3).mean()
    cnt = (((r >= 30) & (r <= 60)).astype(int) + (vol3 < volprec3*0.85).astype(int) +
           (((h-l)/c*100) < range_max).astype(int) + sopra.astype(int))
    setup2 = (cnt >= 3) & (cnt >= 3).shift(1)
    corpo = (c-o).abs()/(h-l).replace(0, np.nan)*100
    breakout = (c > o) & (corpo >= 60) & (v >= v.rolling(20).mean()*1.5)
    return ((setup2 | breakout) & sopra).fillna(False)


GRIGLIE = {
    "pullback":     {"label": "Profondità calo (%)", "valori": [1.0, 1.5, 2.0, 3.0], "default": 1.5},
    "momentum":     {"label": "ADX minimo",          "valori": [15, 20, 25, 30],     "default": 20},
    "compressione": {"label": "Range max (%)",        "valori": [0.8, 1.2, 1.6, 2.0], "default": 0.8},
}
TRAILS = [2.0, 2.5, 3.0, 3.5]   # paletti da swing ATTIVO (operare spesso)
TRAIL_DEFAULT = 3.0


def _build_seg(entry, df, pval):
    if entry == "pullback":
        return seg_pullback_opt(df, pb_prof=pval)
    if entry == "momentum":
        return seg_momentum_opt(df, adx_min=pval)
    return seg_compress_opt(df, range_max=pval)


def ottimizza(tickers, entry="pullback", periodo="5y", costo_pct=0.2, stop_atr=1.5, rischio_pct=0.01):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    slices = []
    for df in dati.values():
        taglio = len(df) // 2
        slices.append((df.iloc[:taglio], df.iloc[taglio:]))

    g = GRIGLIE[entry]
    risultati = []
    for pval in g["valori"]:
        for trail in TRAILS:
            cfg = dict(modo="trailing", trail_atr=trail, stop_atr=stop_atr,
                       rischio_pct=rischio_pct, costo_pct=costo_pct, target_pct=4.0, max_giorni=5)
            tr, te = [], []
            for dtr, dte in slices:
                try:
                    tr += backtest_pnl(dtr, _build_seg(entry, dtr, pval), cfg)["trade"]
                    te += backtest_pnl(dte, _build_seg(entry, dte, pval), cfg)["trade"]
                except Exception:
                    pass
            risultati.append({
                "pval": pval, "trail": trail,
                "exp_train": float(np.mean(tr)/10000*100) if tr else float("nan"),
                "exp_test": float(np.mean(te)/10000*100) if te else float("nan"),
                "n_train": len(tr), "n_test": len(te),
                "is_default": (pval == g["default"] and trail == TRAIL_DEFAULT),
            })

    validi = [r for r in risultati if r["n_train"] >= 30 and not np.isnan(r["exp_train"])]
    best = max(validi, key=lambda r: r["exp_train"]) if validi else None
    default = next((r for r in risultati if r["is_default"]), None)
    return {"risultati": risultati, "best": best, "default": default, "label": g["label"],
            "valori": g["valori"], "trails": TRAILS, "entry": entry,
            "falliti": falliti, "n_titoli": len(dati), "costo_pct": costo_pct}


# ==========================================================
# STRUMENTO OPERATIVO: cruscotto segnali di OGGI + diario
# ==========================================================
# parametri ufficiali decisi insieme (tarati e verificati fuori campione)
PARAMS_UFFICIALI = {
    "pullback":     {"pb_prof": 1.5, "trail": 2.5},
    "momentum":     {"adx_min": 25,  "trail": 2.5},
    "compressione": {"range_max": 1.2, "trail": 2.5},
}


def _seg_for(entry, df):
    if entry == "pullback":
        return seg_pullback_opt(df, pb_prof=PARAMS_UFFICIALI["pullback"]["pb_prof"])
    if entry == "momentum":
        return seg_momentum_opt(df, adx_min=PARAMS_UFFICIALI["momentum"]["adx_min"])
    return seg_compress_opt(df, range_max=PARAMS_UFFICIALI["compressione"]["range_max"])


def segnali_oggi(tickers, assegnazioni, periodo="1y"):
    """
    Per ogni titolo guarda l'ULTIMA candela: c'e' un segnale di entrata oggi?
    assegnazioni: {ticker: "pullback"/"momentum"/"compressione"}
    Restituisce la lista dei titoli che segnalano OGGI + un quadro di tutti.
    """
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []
    for t, df in dati.items():
        entry = assegnazioni.get(t, "pullback")
        try:
            seg = _seg_for(entry, df)
            ultimo = bool(seg.iloc[-1])
            # da quante candele non segnalava (per capire se e' "fresco")
            recenti = seg.iloc[-10:]
            data_ultima = df.index[-1]
            prezzo = float(df["close"].iloc[-1])
            atr_val = float(atr(df, 14).iloc[-1])
            righe.append({
                "ticker": t, "algoritmo": entry, "segnale_oggi": ultimo,
                "data": str(pd.Timestamp(data_ultima).date()),
                "prezzo": round(prezzo, 2),
                "stop_suggerito": round(prezzo - 1.5 * atr_val, 2),
                "atr": round(atr_val, 2),
            })
        except Exception:
            falliti.append(t)
    oggi = [r for r in righe if r["segnale_oggi"]]
    return {"oggi": oggi, "tutti": righe, "falliti": falliti,
            "data_aggiornamento": str(pd.Timestamp.today().date())}


def aggiorna_esito_voce(ticker, prezzo_in, data_in, giorni_max=5, target_pct=3.0):
    """
    Per una voce del diario: scarica i prezzi DOPO la data del segnale e calcola
    com'e' andata (target colpito? quanto su/giu? ancora in corso?).
    """
    import yfinance as yf
    try:
        df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df[["high", "low", "close"]].dropna()
    except Exception:
        return {"stato": "dati non disponibili"}

    d0 = pd.Timestamp(data_in)
    dopo = df[df.index > d0]
    if len(dopo) == 0:
        return {"stato": "in attesa (nessun giorno dopo il segnale)"}

    finestra = dopo.iloc[:giorni_max]
    target = prezzo_in * (1 + target_pct / 100)
    max_h = float(finestra["high"].max())
    min_l = float(finestra["low"].min())
    ultimo = float(finestra["close"].iloc[-1])
    giorni_passati = len(finestra)

    tocco_target = max_h >= target
    var_max = (max_h / prezzo_in - 1) * 100
    var_min = (min_l / prezzo_in - 1) * 100
    var_ora = (ultimo / prezzo_in - 1) * 100

    if giorni_passati < giorni_max:
        stato = f"in corso ({giorni_passati}/{giorni_max} giorni)"
    else:
        stato = "chiuso"

    esito = "🎯 target raggiunto" if tocco_target else ("🟢 in utile" if var_ora > 0 else "🔴 in perdita")
    return {"stato": stato, "esito": esito,
            "var_ora_pct": round(var_ora, 2), "max_pct": round(var_max, 2),
            "min_pct": round(var_min, 2), "giorni": giorni_passati,
            "tocco_target": tocco_target}


# ==========================================================
# PAPER TRADING: simulazione vera con SL/TP fissi o trailing %
# ==========================================================
def _chiudi_pos(pos, prezzo_out, esito, data_out, commissione):
    qty = pos["qty"]; entry = pos["prezzo_in"]
    pnl = qty * (prezzo_out - entry) - pos.get("commissione_in", 0.0) - commissione
    out = dict(pos)
    out.update({"prezzo_out": round(float(prezzo_out), 4), "esito": esito,
                "data_out": str(pd.Timestamp(data_out).date()), "commissione_out": commissione,
                "pnl": round(pnl, 2), "var_pct": round((prezzo_out/entry - 1) * 100, 2)})
    return ("chiusa", out)


def simula_posizione(pos, df, commissione):
    """Cammina sui prezzi dopo l'entrata e decide: stop, target, trailing o ancora aperta."""
    d0 = pd.Timestamp(pos["data_in"])
    dopo = df[df.index > d0]
    entry = pos["prezzo_in"]
    if pos["gestione"] == "trailing":
        tp = pos["trail_pct"] / 100.0
        trail = pos.get("trail", entry * (1 - tp))
        for dt, row in dopo.iterrows():
            trail = max(trail, row["high"] * (1 - tp))   # il paletto sale, mai giu'
            if row["low"] <= trail:
                return _chiudi_pos(pos, trail, "🔻 trailing", dt, commissione)
        prezzo_ora = float(dopo["close"].iloc[-1]) if len(dopo) else entry
        pos["trail"] = round(trail, 4); pos["prezzo_ora"] = round(prezzo_ora, 4); pos["giorni"] = len(dopo)
        return ("aperta", pos)
    else:
        stop, target = pos["stop"], pos["target"]
        for dt, row in dopo.iterrows():
            if row["low"] <= stop:
                return _chiudi_pos(pos, stop, "🛑 stop", dt, commissione)
            if row["high"] >= target:
                return _chiudi_pos(pos, target, "🎯 target", dt, commissione)
        prezzo_ora = float(dopo["close"].iloc[-1]) if len(dopo) else entry
        pos["prezzo_ora"] = round(prezzo_ora, 4); pos["giorni"] = len(dopo)
        return ("aperta", pos)


def aggiorna_portafoglio(port):
    """Aggiorna tutte le posizioni aperte: chiude quelle che hanno toccato stop/target/trailing."""
    import yfinance as yf
    comm = port.get("commissione", 0.99)
    ancora, chiuse_nuove = [], []
    for pos in port["aperte"]:
        try:
            df = yf.download(pos["ticker"], period="6mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            df = df[["high", "low", "close"]].dropna()
        except Exception:
            ancora.append(pos); continue
        stato, out = simula_posizione(pos, df, comm)
        if stato == "chiusa":
            port["contante"] = round(port["contante"] + out["qty"] * out["prezzo_out"] - comm, 2)
            chiuse_nuove.append(out)
        else:
            ancora.append(out)
    port["aperte"] = ancora
    port["chiuse"] = port.get("chiuse", []) + chiuse_nuove
    return port, len(chiuse_nuove)


def valore_portafoglio(port):
    aperte_val = sum(p["qty"] * p.get("prezzo_ora", p["prezzo_in"]) for p in port["aperte"])
    totale = port["contante"] + aperte_val
    pnl = totale - port["capitale_iniziale"]
    return {"contante": round(port["contante"], 2), "aperte_val": round(aperte_val, 2),
            "totale": round(totale, 2), "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / port["capitale_iniziale"] * 100, 2)}


# ==========================================================
# ANAGRAFICA TITOLI (lista salvata) + azioni INTERE
# ==========================================================
def calcola_azioni_intere(importo_voluto, prezzo):
    """Quante azioni INTERE entrano nell'importo. Restituisce (qty, importo_reale)."""
    if prezzo <= 0:
        return 0, 0.0
    qty = int(importo_voluto // prezzo)        # solo azioni intere (come i broker reali)
    return qty, round(qty * prezzo, 2)


def prezzo_corrente(ticker):
    """Ultimo prezzo di chiusura disponibile per un titolo."""
    import yfinance as yf
    df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato per '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return float(df["close"].dropna().iloc[-1])


# ==========================================================
# ANAGRAFICA TITOLI con classificazione automatica del carattere
#   calmo  -> Momentum ; nervoso -> Pullback
#   (la soglia di volatilita' annua che separa i due mondi)
# ==========================================================
SOGLIA_NERVOSO = 40.0   # volatilita' annua %: sopra = nervoso, sotto = calmo


def classifica_titolo(ticker, periodo="2y"):
    """Misura la volatilita' del titolo e dice se e' calmo (Momentum) o nervoso (Pullback)."""
    import yfinance as yf
    df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Nessun dato per '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if len(df) < 60:
        raise ValueError(f"Storico troppo corto per '{ticker}'.")
    vol = volatilita_annua(df)
    algoritmo = "pullback" if vol >= SOGLIA_NERVOSO else "momentum"
    return {"ticker": ticker, "volatilita": round(vol, 1),
            "carattere": "nervoso" if algoritmo == "pullback" else "calmo",
            "algoritmo": algoritmo, "prezzo": round(float(df["close"].iloc[-1]), 2)}


# ---- acquisto ad AZIONI INTERE (come fanno i broker reali) ----
def azioni_intere(importo_voluto, prezzo):
    """Quante azioni INTERE entrano nell'importo voluto, e l'importo reale."""
    if prezzo <= 0:
        return 0, 0.0
    n = int(importo_voluto // prezzo)
    return n, round(n * prezzo, 2)


# ==========================================================
# COMPARATORE A CONTI PARALLELI: 3 conti gemelli, 1 per algoritmo
#   stessa watchlist, stesso capitale, stesso periodo, gestione trailing,
#   commissioni e azioni intere reali. Simulazione automatica retrospettiva.
# ==========================================================
def _simula_conto(dati, entry, capitale, commissione, importo, trail_pct):
    """Apre/chiude in automatico le posizioni di UN algoritmo su tutti i titoli."""
    cassa = capitale
    chiuse = []
    aperte_finali = []
    curva = []  # valore totale nel tempo (approssimato a fine periodo per titolo)
    for t, df in dati.items():
        seg = _seg_for(entry, df)
        r = df.reset_index()
        col_data = r.columns[0]
        op, hi, lo, cl = r["open"].values, r["high"].values, r["low"].values, r["close"].values
        n = len(r)
        in_pos = False
        entry_px = qty = trail = 0.0
        d_in = None
        i = 0
        while i < n - 1:
            if not in_pos:
                if bool(seg.iloc[i]):
                    px = op[i+1]
                    naz, imp = azioni_intere(importo, px)
                    if naz >= 1 and (imp + commissione) <= cassa:
                        entry_px = px; qty = naz
                        cassa -= imp + commissione
                        trail = entry_px * (1 - trail_pct/100)
                        d_in = r[col_data].iloc[i+1]
                        in_pos = True
            else:
                trail = max(trail, hi[i] * (1 - trail_pct/100))
                if lo[i] <= trail:
                    cassa += qty * trail - commissione
                    chiuse.append({"ticker": t, "entrata": round(entry_px, 2), "uscita": round(trail, 2),
                                   "qty": qty, "pnl": round(qty*(trail-entry_px) - 2*commissione, 2),
                                   "var_pct": round((trail/entry_px-1)*100, 2)})
                    in_pos = False
            i += 1
        if in_pos:   # chiusura a fine periodo al prezzo finale
            px = cl[n-1]
            cassa += qty * px - commissione
            chiuse.append({"ticker": t, "entrata": round(entry_px, 2), "uscita": round(px, 2),
                           "qty": qty, "pnl": round(qty*(px-entry_px) - 2*commissione, 2),
                           "var_pct": round((px/entry_px-1)*100, 2), "esito": "fine"})
    n_tr = len(chiuse)
    vinte = sum(1 for x in chiuse if x["pnl"] > 0)
    pnl_tot = round(sum(x["pnl"] for x in chiuse), 2)
    costi = round(n_tr * 2 * commissione, 2)
    return {"capitale_finale": round(cassa, 2), "pnl": pnl_tot,
            "pnl_pct": round((cassa/capitale - 1) * 100, 2),
            "n_trade": n_tr, "vinte": vinte,
            "win_rate": round(vinte/n_tr*100, 1) if n_tr else 0.0,
            "costi": costi, "operazioni": chiuse}


def confronta_conti_paralleli(tickers, periodo="2y", capitale=600.0, commissione=0.99,
                              importo=100.0, trail_pct=8.0):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    risultati = {}
    for nome, entry in [("Momentum", "momentum"), ("Pullback", "pullback"), ("Compressione", "compressione")]:
        risultati[nome] = _simula_conto(dati, entry, capitale, commissione, importo, trail_pct)
    return {"risultati": risultati, "n_titoli": len(dati), "falliti": falliti,
            "capitale": capitale, "periodo": periodo}


# ==========================================================
# SEGNALI DI OGGI con TUTTI E TRE gli algoritmi su ogni titolo
# ==========================================================
def segnali_oggi_multi(tickers, periodo="2y"):
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []
    for t, df in dati.items():
        try:
            prezzo = float(df["close"].iloc[-1])
            atr_val = float(atr(df, 14).iloc[-1])
            s = {}
            for entry in ("momentum", "pullback", "compressione"):
                try:
                    s[entry] = bool(_seg_for(entry, df).iloc[-1])
                except Exception:
                    s[entry] = False
            righe.append({"ticker": t, "prezzo": round(prezzo, 2),
                          "stop_suggerito": round(prezzo - 1.5 * atr_val, 2),
                          "data": str(pd.Timestamp(df.index[-1]).date()),
                          "momentum": s["momentum"], "pullback": s["pullback"],
                          "compressione": s["compressione"], "qualcuno": any(s.values())})
        except Exception:
            falliti.append(t)
    return {"righe": righe, "falliti": falliti,
            "data_aggiornamento": str(pd.Timestamp.today().date())}


# ==========================================================
# STORICO AUTOMATICO DEI SEGNALI (archivio-laboratorio)
#   registra TUTTI i segnali di oggi (no doppioni) e ne calcola l'esito
#   con regole stop/take FISSE, diverse per calmi e nervosi.
# ==========================================================
def chiave_segnale(s):
    return f"{s['ticker']}|{s['data']}|{s['algoritmo']}"


def raccogli_segnali_oggi(tickers, carattere_map, periodo="2y"):
    """Tutti i segnali (tutti e 3 gli algoritmi) accesi OGGI, pronti per l'archivio."""
    multi = segnali_oggi_multi(tickers, periodo)
    nuovi = []
    for r in multi["righe"]:
        for algo in ("momentum", "pullback", "compressione"):
            if r[algo]:
                nuovi.append({"ticker": r["ticker"], "data": r["data"], "algoritmo": algo,
                              "prezzo_in": r["prezzo"],
                              "carattere": carattere_map.get(r["ticker"], "calmo")})
    return {"segnali": nuovi, "falliti": multi["falliti"],
            "data_aggiornamento": multi["data_aggiornamento"]}


def esito_segnale_fisso(ticker, prezzo_in, data_in, stop_pct, take_pct, giorni_max=20):
    """Calcola l'esito di UN segnale con stop/take fissi (in %). Onesto: stop prima del target."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df[["high", "low", "close"]].dropna()
    except Exception:
        return {"stato": "dati n/d"}
    dopo = df[df.index > pd.Timestamp(data_in)]
    if len(dopo) == 0:
        return {"stato": "in attesa", "esito": "—"}
    finestra = dopo.iloc[:giorni_max]
    stop = prezzo_in * (1 - stop_pct/100)
    target = prezzo_in * (1 + take_pct/100)
    for dt, row in finestra.iterrows():
        if row["low"] <= stop:
            return {"stato": "chiuso", "esito": "🛑 stop", "var_pct": round(-stop_pct, 2),
                    "giorni": int((finestra.index <= dt).sum()), "data_out": str(dt.date())}
        if row["high"] >= target:
            return {"stato": "chiuso", "esito": "🎯 target", "var_pct": round(take_pct, 2),
                    "giorni": int((finestra.index <= dt).sum()), "data_out": str(dt.date())}
    ultimo = float(finestra["close"].iloc[-1])
    var_ora = (ultimo / prezzo_in - 1) * 100
    chiuso = len(finestra) >= giorni_max
    return {"stato": "chiuso (tempo)" if chiuso else f"in corso ({len(finestra)}g)",
            "esito": "⏰ tempo" if chiuso else "🟢 aperto" if var_ora >= 0 else "🔴 aperto",
            "var_pct": round(var_ora, 2), "giorni": len(finestra)}


# ==========================================================
# MONITOR DIURNO: segnali ricalcolati col prezzo di ADESSO
#   (a mercato aperto, per intercettare i gap da notizie notturne)
# ==========================================================
def prezzo_intraday(ticker):
    """Ultimo prezzo disponibile a mercato aperto (Yahoo, ~15 min di ritardo)."""
    import yfinance as yf
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        riga = df.dropna(subset=["close"]).iloc[-1]
        # precisione piena: l'arrotondamento si fa solo in visualizzazione,
        # altrimenti su titoli sotto l'euro il calcolo del gap si sporca
        return {"prezzo": float(riga["close"]),
                "ora_dato": str(pd.Timestamp(df.index[-1]))}
    except Exception:
        return None


def segnali_diurni(tickers, carattere_map, periodo="2y"):
    """
    Segnali ricalcolati sostituendo l'ULTIMA candela giornaliera con i prezzi
    intraday di adesso: cosi' il segnale tiene conto del gap di apertura.
    Confronta anche col segnale 'serale' (sui dati di chiusura) per vedere cos'e' cambiato.
    """
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []
    for t, df in dati.items():
        try:
            # segnale "serale" = sull'ultima chiusura nota
            sera = {a: bool(_seg_for(a, df).iloc[-1]) for a in ("momentum", "pullback", "compressione")}
            prezzo_sera = round(float(df["close"].iloc[-1]), 2)

            # segnale "diurno" = aggiorno l'ultima candela col prezzo di adesso
            intr = prezzo_intraday(t)
            if intr:
                df2 = df.copy()
                px = intr["prezzo"]
                i = df2.index[-1]
                df2.loc[i, "close"] = px
                df2.loc[i, "high"] = max(df2.loc[i, "high"], px)
                df2.loc[i, "low"] = min(df2.loc[i, "low"], px)
                giorno = {a: bool(_seg_for(a, df2).iloc[-1]) for a in ("momentum", "pullback", "compressione")}
                prezzo_ora = px
                ora_dato = intr["ora_dato"]
            else:
                giorno = sera
                prezzo_ora = prezzo_sera
                ora_dato = "n/d"

            n_attivi = sum(giorno.values())
            cambiato = giorno != sera
            righe.append({"ticker": t, "carattere": carattere_map.get(t, "calmo"),
                          "prezzo_ora": prezzo_ora, "prezzo_sera": prezzo_sera,
                          "gap_pct": round((prezzo_ora/prezzo_sera - 1) * 100, 2) if prezzo_sera else 0.0,
                          "momentum": giorno["momentum"], "pullback": giorno["pullback"],
                          "compressione": giorno["compressione"],
                          "n_attivi": n_attivi, "confluenza": n_attivi >= 2,
                          "qualcuno": n_attivi >= 1, "cambiato_da_sera": cambiato,
                          "ora_dato": ora_dato})
        except Exception:
            falliti.append(t)
    return {"righe": righe, "falliti": falliti,
            "ora_controllo": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))}



# ==========================================================
# CONTROLLO DEL MATTINO (anti-gap)
#   Filosofia: il SEGNALE resta quello calcolato sulla candela
#   COMPLETA di ieri (affidabile). Al mattino non si ricalcola
#   nulla sulla candela viva: si verifica solo se il gap di
#   apertura ha invalidato il segnale, con un verdetto meccanico.
#
#   Verdetti (misurati in unita' di ATR di ieri):
#     OK             gap contenuto: il piano di ieri sera vale ancora
#     NON INSEGUIRE  gap su oltre soglia: si entrerebbe a un prezzo
#                    che il backtest non ha mai pagato
#     ANNULLATO      gap giu' oltre soglia: il presupposto del
#                    segnale e' saltato (spesso c'e' una notizia)
#
#   I verdetti sono etichette meccaniche delle condizioni,
#   non consigli operativi: la decisione resta dell'utente.
# ==========================================================
def controllo_mattino(tickers, periodo="2y",
                      soglia_ok_atr=0.5, soglia_annulla_atr=0.5):
    """
    Per ogni titolo:
      1) segnali calcolati sull'ultima candela GIORNALIERA COMPLETA
         (se i dati includono la candela di oggi in corso, viene scartata)
      2) prezzo di adesso (intraday ~15 min di ritardo) e gap % vs ieri
      3) verdetto OK / NON INSEGUIRE / ANNULLATO in base al gap in ATR
    Restituisce solo numeri grezzi: l'arrotondamento e' della UI.
    """
    oggi = pd.Timestamp.today().normalize()
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []

    for t, df in dati.items():
        try:
            # 1) tieni SOLO candele complete: via quella di oggi se presente
            comp = df[df.index.normalize() < oggi]
            if len(comp) < 250:
                falliti.append(t)
                continue

            seg = {}
            for entry in ("momentum", "pullback", "compressione"):
                try:
                    seg[entry] = bool(_seg_for(entry, comp).iloc[-1])
                except Exception:
                    seg[entry] = False
            n_attivi = sum(seg.values())

            chiusura_ieri = float(comp["close"].iloc[-1])
            atr_ieri = float(atr(comp, 14).iloc[-1])
            atr_pct = atr_ieri / chiusura_ieri * 100 if chiusura_ieri else 0.0

            # 2) prezzo di adesso (se non disponibile: mercato chiuso o
            #    dato mancante -> gap non valutabile, si segnala)
            intr = prezzo_intraday(t)
            if intr and intr["prezzo"]:
                prezzo_ora = float(intr["prezzo"])
                ora_dato = intr["ora_dato"]
                gap_pct = (prezzo_ora / chiusura_ieri - 1) * 100
                gap_atr = gap_pct / atr_pct if atr_pct > 0 else 0.0
            else:
                prezzo_ora, ora_dato = chiusura_ieri, "n/d"
                gap_pct, gap_atr = 0.0, 0.0

            # 3) verdetto meccanico (solo se ieri sera c'era un segnale)
            if n_attivi == 0:
                verdetto = "—"
            elif intr is None:
                verdetto = "GAP N/D"
            elif gap_atr > soglia_ok_atr:
                verdetto = "NON INSEGUIRE"
            elif gap_atr < -soglia_annulla_atr:
                verdetto = "ANNULLATO"
            else:
                verdetto = "OK"

            righe.append({
                "ticker": t,
                "momentum": seg["momentum"], "pullback": seg["pullback"],
                "compressione": seg["compressione"],
                "n_attivi": n_attivi, "confluenza": n_attivi >= 2,
                "qualcuno": n_attivi >= 1,
                "data_segnale": str(pd.Timestamp(comp.index[-1]).date()),
                "chiusura_ieri": chiusura_ieri,
                "prezzo_ora": prezzo_ora, "ora_dato": ora_dato,
                "gap_pct": gap_pct, "gap_atr": gap_atr,
                "atr_pct": atr_pct,
                # stop indicativo dall'ATR di IERI (candela completa),
                # ancorato al prezzo di adesso
                "stop_indicativo": prezzo_ora - 1.5 * atr_ieri,
                "verdetto": verdetto,
            })
        except Exception:
            falliti.append(t)

    return {"righe": righe, "falliti": falliti,
            "ora_controllo": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "soglia_ok_atr": soglia_ok_atr, "soglia_annulla_atr": soglia_annulla_atr}


# ==========================================================
# SEMAFORO D'INGRESSO (qualsiasi ora di mercato aperto)
#   Evoluzione del controllo del mattino. Combina:
#     1) segnali di IERI SERA su candela completa (+ confluenze)
#     2) TRIGGER: il massimo di ieri deve essere superato OGGI
#        (conferma di prezzo: e' il mercato a dire che parte)
#     3) ESTENSIONE: quanto e' gia' corso oltre il trigger
#        (per non inseguire un ingresso che il backtest non ha mai pagato)
#
#   Stati (etichette meccaniche, la decisione resta dell'utente):
#     VIA LIBERA      trigger superato, prezzo ancora vicino: il piano vale
#     ATTESA TRIGGER  segnale valido ma il massimo di ieri non e' stato rotto
#     NON INSEGUIRE   trigger superato ma prezzo gia' esteso oltre soglia
#     RIENTRATO       trigger toccato oggi ma prezzo ricaduto sotto:
#                     rottura fallita, segnale di debolezza
#     ANNULLATO       prezzo sotto la chiusura di ieri oltre soglia:
#                     presupposto saltato
#     DATI N/D        intraday non disponibile (mercato chiuso?)
# ==========================================================
def quadro_intraday(ticker):
    """Prezzo attuale E massimo di OGGI (5m, ~15 min di ritardo).
    Il massimo di oggi serve per sapere se il trigger e' stato toccato
    anche quando il prezzo, adesso, e' rientrato sotto."""
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
        oggi = df[df.index.normalize() == df.index[-1].normalize()]
        if oggi.empty:
            return None
        return {"prezzo": float(oggi["close"].iloc[-1]),
                "massimo_oggi": float(oggi["high"].max()),
                "ora_dato": str(pd.Timestamp(df.index[-1]))}
    except Exception:
        return None


def semaforo_ingresso(tickers, periodo="2y",
                      soglia_estensione_atr=1.0, soglia_annulla_atr=0.5):
    """
    Per ogni titolo con segnale IERI SERA (candela completa):
      trigger = massimo di ieri; si confronta con massimo e prezzo di OGGI.
    Restituisce numeri grezzi; l'arrotondamento e' della UI.
    """
    oggi = pd.Timestamp.today().normalize()
    dati = carica_watchlist(tickers, periodo)
    falliti = [t for t in tickers if t and t not in dati]
    righe = []

    for t, df in dati.items():
        try:
            comp = df[df.index.normalize() < oggi]   # solo candele complete
            if len(comp) < 250:
                falliti.append(t)
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

            q = quadro_intraday(t) if n_attivi else None
            prezzo_ora = q["prezzo"] if q else chiusura_ieri
            massimo_oggi = q["massimo_oggi"] if q else None
            ora_dato = q["ora_dato"] if q else "n/d"

            # distanze utili (in ATR di ieri)
            dist_trigger_atr = ((prezzo_ora - trigger) / atr_ieri) if atr_ieri else 0.0
            dist_chiusura_atr = ((prezzo_ora - chiusura_ieri) / atr_ieri) if atr_ieri else 0.0

            # --- stato, in ordine di priorita' ---
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
                "data_segnale": str(pd.Timestamp(comp.index[-1]).date()),
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
        except Exception:
            falliti.append(t)

    return {"righe": righe, "falliti": falliti,
            "ora_controllo": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "soglia_estensione_atr": soglia_estensione_atr,
            "soglia_annulla_atr": soglia_annulla_atr}
