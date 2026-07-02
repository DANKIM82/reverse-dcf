#!/usr/bin/env python3
"""역DCF 유니버스 스크리너 - FinanceDataReader + NAVER 재무 파이프라인"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

SECTOR_WACC = {
    "반도체": 0.105, "IT부품": 0.105, "2차전지": 0.11, "인터넷": 0.105,
    "게임": 0.11, "자동차": 0.095, "조선": 0.10, "방산": 0.095,
    "철강화학": 0.095, "바이오": 0.115, "음식료": 0.08, "화장품": 0.095,
    "금융": 0.09, "지주": 0.09, "유틸통신": 0.075, "엔터": 0.11,
    "유통소비": 0.085, "건설기계": 0.10, "운송": 0.10, "기타": 0.095,
}
DEFAULT_TERMINAL_G = 0.01

# DEMO_UNIVERSE에 있는 EV/FCF hint (negative = 적자)
EVFCF_HINT = {
    "005930": 18, "000660": 14, "042700": 38, "058470": 26, "403870": 30, "240810": 22,
    "011070": 9, "009150": 15, "007660": 34, "373220": -1, "006400": -1, "247540": -1,
    "003670": 45, "035420": 17, "035720": 24, "259960": 13, "036570": 12, "251270": 28,
    "005380": 7, "000270": 6, "012330": 8, "018880": 16, "161390": 7, "009540": 20,
    "010140": 25, "042660": -1, "329180": 28, "012450": 30, "079550": 27, "064350": 26,
    "005490": 9, "010130": 15, "011170": -1, "011780": 8, "051910": 40, "207940": 42,
    "068270": 35, "000100": 35, "196170": 55, "097950": 8, "271560": 9, "004370": 9,
    "033780": 10, "090430": 22, "051900": 12, "192820": 13, "161890": 12, "257720": 24,
    "105560": 6, "055550": 6, "086790": 5, "316140": 5, "138040": 8, "000810": 8,
    "005830": 6, "323410": 15, "034730": 5, "003550": 7, "028260": 11, "000880": 4,
    "006260": 7, "001040": 5, "000150": 12, "015760": -1, "030200": 6, "017670": 7,
    "032640": 5, "352820": 30, "035900": 14, "041510": 15, "122870": 18, "021240": 9,
    "139480": -1, "282330": 8, "007070": 7, "008770": 20, "000720": 7, "034020": 48,
    "241560": 6, "267260": 32, "010120": 25, "298040": 28, "003490": 7, "011200": 5,
    "086280": 8,
}


def solve_implied_growth(ev, fcf0, wacc, tg, n=5, lo=-0.5, hi=1.0):
    if fcf0 is None or fcf0 <= 0 or ev is None or ev <= 0:
        return np.nan, "fcf<=0"
    if wacc <= tg:
        return np.nan, "wacc<=tg"

    def calc_ev(g):
        t = np.arange(1, n + 1)
        fcf = fcf0 * (1 + g) ** t
        pv = np.sum(fcf / (1 + wacc) ** t)
        tv = fcf[-1] * (1 + tg) / (wacc - tg)
        return pv + tv / (1 + wacc) ** n

    f_lo, f_hi = calc_ev(lo) - ev, calc_ev(hi) - ev
    if f_lo > 0:
        return np.nan, "ev_too_low"
    if f_hi < 0:
        return np.nan, "ev_too_high"
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if calc_ev(mid) - ev > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), "ok"


def fetch_naver_json_financials(code, session):
    """NAVER 금융 JSON API로 현금흐름 데이터 가져오기"""
    empty = {"fcf_ttm": None, "fcf_3y_avg": None, "revenue_ttm": None,
              "norm_fcf_margin": None, "fcf_cagr_5y": None, "net_debt": 0.0}
    try:
        # NAVER 재무제표 - 현금흐름표 JSON API
        url = "https://finance.naver.com/item/coinfo.naver?code=" + code + "&target=cashflow"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://finance.naver.com/",
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        r = session.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return empty

        import pandas as pd
        from io import StringIO
        tables = pd.read_html(StringIO(r.text), header=0)
        if not tables:
            return empty

        cfo_list, capex_list, rev_list = [], [], []
        td_list, ltd_list, cash_list = [], [], []

        for tbl in tables:
            tbl.columns = [str(c).strip() for c in tbl.columns]
            # 숫자 컬럼 추출 (연도별 데이터)
            num_cols = []
            for c in tbl.columns[1:]:
                try:
                    if str(c).replace("/", "").replace(".", "").isdigit() or "12" in str(c) or "연간" in str(c):
                        num_cols.append(c)
                except Exception:
                    pass
            if not num_cols:
                num_cols = list(tbl.columns[1:min(6, len(tbl.columns))])

            for idx, row in tbl.iterrows():
                label = str(row.iloc[0]).replace(" ", "").lower() if len(row) > 0 else ""
                vals = []
                for c in num_cols:
                    try:
                        v = str(row[c]).replace(",", "").strip()
                        if v not in ["nan", "-", "", "N/A"]:
                            fv = float(v)
                            vals.append(fv)
                    except Exception:
                        pass
                if not vals:
                    continue

                if any(k in label for k in ["영업활동으로인한현금흐름", "영업활동현금흐름", "operatingcash"]):
                    cfo_list = vals
                elif any(k in label for k in ["유형자산의취득", "유형자산취득", "capex"]):
                    capex_list = vals
                elif any(k in label for k in ["매출액", "순영업수익", "revenue"]):
                    if not rev_list:
                        rev_list = vals
                elif "단기차입금" in label:
                    td_list = vals
                elif any(k in label for k in ["장기차입금", "사채"]):
                    ltd_list = vals
                elif "현금및현금성자산" in label and not cash_list:
                    cash_list = vals

        if not cfo_list:
            return empty

        # 억원 단위로 통일 (원 단위면 변환)
        def normalize(vals):
            if vals and abs(vals[0]) > 1e8:
                return [v / 1e8 for v in vals]
            return vals

        cfo_list = normalize(cfo_list)
        capex_list = normalize(capex_list) if capex_list else []
        rev_list = normalize(rev_list) if rev_list else []
        td_list = normalize(td_list) if td_list else []
        ltd_list = normalize(ltd_list) if ltd_list else []
        cash_list = normalize(cash_list) if cash_list else []

        fcf_series = []
        for i, cfo in enumerate(cfo_list):
            capex = capex_list[i] if i < len(capex_list) else 0.0
            fcf_series.append(cfo - abs(capex))

        fcf_ttm = fcf_series[-1] if fcf_series else None
        fcf_3y = float(np.mean(fcf_series[-3:])) if len(fcf_series) >= 3 else fcf_ttm
        cagr = None
        if len(fcf_series) >= 4 and fcf_series[0] > 0 and fcf_series[-1] > 0:
            cagr = (fcf_series[-1] / fcf_series[0]) ** (1 / (len(fcf_series) - 1)) - 1

        rev_ttm = rev_list[-1] if rev_list else None
        margin = None
        if rev_ttm and rev_ttm > 0 and fcf_series:
            pos = [v for v in fcf_series if v > 0]
            if pos:
                margin = min(max(float(np.median(pos)) / rev_ttm, 0.01), 0.35)

        td = td_list[-1] if td_list else 0.0
        ltd = ltd_list[-1] if ltd_list else 0.0
        cash = cash_list[-1] if cash_list else 0.0
        net_debt = td + ltd - cash

        r2 = lambda x, d=1: None if x is None else round(float(x), d)
        return {
            "fcf_ttm": r2(fcf_ttm),
            "fcf_3y_avg": r2(fcf_3y),
            "revenue_ttm": r2(rev_ttm),
            "norm_fcf_margin": r2(margin, 4),
            "fcf_cagr_5y": r2(cagr, 4),
            "net_debt": r2(net_debt) or 0.0,
        }
    except Exception as e:
        print("    naver_fin(" + code + "): " + str(e))
        return empty


def fetch_live(universe_csv):
    """FinanceDataReader 시총 + NAVER 재무 수집"""
    import pandas as pd
    import requests
    import FinanceDataReader as fdr

    uni = pd.read_csv(universe_csv, dtype={"code": str}, encoding="utf-8-sig")
    names = []

    print("KRX 시가총액 전체 다운로드 중...")
    cap_all = None
    try:
        cap_all = fdr.StockListing('KRX')
        cap_all.columns = [c.strip() for c in cap_all.columns]
        for cn in ['Code', 'Symbol']:
            if cn in cap_all.columns:
                cap_all = cap_all.rename(columns={cn: 'Code'})
                break
        cap_all['Code'] = cap_all['Code'].astype(str).str.zfill(6)
        print("  OK: " + str(len(cap_all)) + "종목")
    except Exception as e:
        print("  KRX 리스팅 실패: " + str(e))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
        "Accept-Language": "ko-KR,ko;q=0.9",
    })

    for _, row in uni.iterrows():
        code = row["code"].zfill(6)
        name = row["name"]
        sector = row["sector"]
        holdco = bool(int(row.get("holdco", 0)))
        try:
            mktcap = None
            if cap_all is not None:
                match = cap_all[cap_all['Code'] == code]
                if not match.empty:
                    for col in ['Marcap', 'MarCap', 'marcap', '시가총액', 'Mktcap']:
                        if col in match.columns:
                            v = match.iloc[0][col]
                            if pd.notna(v) and float(v) > 0:
                                raw = float(v)
                                mktcap = raw / 1e8 if raw > 1e10 else raw
                                break
            if mktcap is None or mktcap <= 0:
                print("  " + code + " " + name + " 실패: 시총 없음")
                continue

            fin = fetch_naver_json_financials(code, session)

            # FCF가 null이면 EV/FCF hint로 추정
            if fin["fcf_ttm"] is None:
                hint = EVFCF_HINT.get(code, -1)
                if hint > 0:
                    ev_est = mktcap  # net_debt 모를 때 mktcap으로 근사
                    fcf_est = round(ev_est / hint, 1)
                    fin["fcf_ttm"] = fcf_est
                    fin["fcf_3y_avg"] = fcf_est
                    fin["fcf_cagr_5y"] = 0.05
                    print("    " + code + " FCF hint 사용: " + str(fcf_est) + "억 (hint=" + str(hint) + ")")

            net_debt = 0.0 if sector in ("금융", "지주") else fin["net_debt"]
            names.append({
                "code": code, "name": name, "sector": sector, "holdco": holdco,
                "mktcap": round(mktcap, 1), "net_debt": round(net_debt, 1),
                "ev": round(mktcap + net_debt, 1),
                "fcf_ttm": fin["fcf_ttm"], "fcf_3y_avg": fin["fcf_3y_avg"],
                "revenue_ttm": fin["revenue_ttm"], "norm_fcf_margin": fin["norm_fcf_margin"],
                "fcf_cagr_5y": fin["fcf_cagr_5y"],
            })
            fcf_info = "fcf=" + str(fin["fcf_ttm"]) if fin["fcf_ttm"] else "fcf=null"
            print("  " + code + " " + name + " ok mktcap=" + str(round(mktcap)) + "억 " + fcf_info)
            time.sleep(0.2)
        except Exception as e:
            print("  " + code + " " + name + " 실패: " + str(e))
    return names


DEMO_UNIVERSE = [
    ("005930","삼성전자","반도체",0,400,18),("000660","SK하이닉스","반도체",0,190,14),
    ("042700","한미반도체","반도체",0,12,38),("058470","리노공업","반도체",0,4,26),
    ("403870","HPSP","반도체",0,3,30),("240810","원익IPS","반도체",0,2,22),
    ("011070","LG이노텍","IT부품",0,5,9),("009150","삼성전기","IT부품",0,12,15),
    ("007660","이수페타시스","IT부품",0,3,34),
    ("373220","LG에너지솔루션","2차전지",0,85,-1),("006400","삼성SDI","2차전지",0,25,-1),
    ("247540","에코프로비엠","2차전지",0,12,-1),("003670","포스코퓨처엠","2차전지",0,10,45),
    ("035420","NAVER","인터넷",0,32,17),("035720","카카오","인터넷",0,18,24),
    ("259960","크래프톤","게임",0,16,13),("036570","엔씨소프트","게임",0,4,12),
    ("251270","넷마블","게임",0,4,28),
    ("005380","현대차","자동차",0,50,7),("000270","기아","자동차",0,40,6),
    ("012330","현대모비스","자동차",0,24,8),("018880","한온시스템","자동차",0,4,16),
    ("161390","한국타이어앤테크놀로지","자동차",0,6,7),
    ("009540","HD한국조선해양","조선",0,15,20),("010140","삼성중공업","조선",0,11,25),
    ("042660","한화오션","조선",0,10,-1),("329180","HD현대중공업","조선",0,20,28),
    ("012450","한화에어로스페이스","방산",0,32,30),("079550","LIG넥스원","방산",0,8,27),
    ("064350","현대로템","방산",0,12,26),
    ("005490","POSCO홀딩스","철강화학",0,25,9),("010130","고려아연","철강화학",0,16,15),
    ("011170","롯데케미칼","철강화학",0,3,-1),("011780","금호석유","철강화학",0,4,8),
    ("051910","LG화학","철강화학",0,22,40),
    ("207940","삼성바이오로직스","바이오",0,70,42),("068270","셀트리온","바이오",0,40,35),
    ("000100","유한양행","바이오",0,9,35),("196170","알테오젠","바이오",0,18,55),
    ("097950","CJ제일제당","음식료",0,5,8),("271560","오리온","음식료",0,4,9),
    ("004370","농심","음식료",0,2,9),("033780","KT&G","음식료",0,13,10),
    ("090430","아모레퍼시픽","화장품",0,8,22),("051900","LG생활건강","화장품",0,5,12),
    ("192820","코스맥스","화장품",0,2,13),("161890","한국콜마","화장품",0,2,12),
    ("257720","실리콘투","화장품",0,2,24),
    ("105560","KB금융","금융",0,35,6),("055550","신한지주","금융",0,27,6),
    ("086790","하나금융지주","금융",0,18,5),("316140","우리금융지주","금융",0,12,5),
    ("138040","메리츠금융지주","금융",0,22,8),("000810","삼성화재","금융",0,17,8),
    ("005830","DB손해보험","금융",0,8,6),("323410","카카오뱅크","금융",0,10,15),
    ("034730","SK","지주",1,13,5),("003550","LG","지주",1,12,7),
    ("028260","삼성물산","지주",1,22,11),("000880","한화","지주",1,3,4),
    ("006260","LS","지주",1,4,7),("001040","CJ","지주",1,3,5),
    ("000150","두산","지주",1,6,12),
    ("015760","한국전력","유틸통신",0,15,-1),("030200","KT","유틸통신",0,12,6),
    ("017670","SK텔레콤","유틸통신",0,12,7),("032640","LG유플러스","유틸통신",0,4,5),
    ("352820","하이브","엔터",0,10,30),("035900","JYP Ent.","엔터",0,2,14),
    ("041510","에스엠","엔터",0,2,15),("122870","와이지엔터테인먼트","엔터",0,1,18),
    ("021240","코웨이","유통소비",0,6,9),("139480","이마트","유통소비",0,2,-1),
    ("282330","BGF리테일","유통소비",0,2,8),("007070","GS리테일","유통소비",0,2,7),
    ("008770","호텔신라","유통소비",0,2,20),
    ("000720","현대건설","건설기계",0,4,7),("034020","두산에너빌리티","건설기계",0,15,48),
    ("241560","두산밥캣","건설기계",0,5,6),("267260","HD현대일렉트릭","건설기계",0,14,32),
    ("010120","LS일렉트릭","건설기계",0,6,25),("298040","효성중공업","건설기계",0,5,28),
    ("003490","대한항공","운송",0,9,7),("011200","HMM","운송",0,16,5),
    ("086280","현대글로비스","운송",0,10,8),
]


def export_universe_csv(path):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "sector", "holdco"])
        for code, name, sector, holdco, *_ in DEMO_UNIVERSE:
            w.writerow([code, name, sector, holdco])
    print(str(path) + ": " + str(len(DEMO_UNIVERSE)) + "종목")


def gen_demo(seed=42):
    rng = np.random.default_rng(seed)
    names = []
    for code, name, sector, holdco, cap_hint, evfcf_hint in DEMO_UNIVERSE:
        mktcap = cap_hint * 1e4 * rng.uniform(0.85, 1.15)
        nd_ratio = rng.uniform(-0.15, 0.35)
        if sector == "금융":
            nd_ratio = 0.0
        net_debt = mktcap * nd_ratio
        ev = mktcap + net_debt
        if evfcf_hint < 0:
            fcf_ttm = -mktcap * rng.uniform(0.005, 0.03)
            fcf_3y = fcf_ttm * rng.uniform(0.3, 1.2)
            cagr = None
        else:
            fcf_ttm = ev / (evfcf_hint * rng.uniform(0.85, 1.2))
            fcf_3y = fcf_ttm * rng.uniform(0.8, 1.15)
            cagr = float(np.clip(rng.normal(0.05, 0.09), -0.20, 0.35))
        rev = mktcap * rng.uniform(0.4, 1.6)
        margin = float(np.clip(abs(fcf_3y) / rev * rng.uniform(0.9, 1.3), 0.02, 0.25))
        r = lambda x, d=1: None if x is None else round(float(x), d)
        names.append({
            "code": code, "name": name, "sector": sector, "holdco": bool(holdco),
            "mktcap": r(mktcap), "net_debt": r(net_debt), "ev": r(ev),
            "fcf_ttm": r(fcf_ttm), "fcf_3y_avg": r(fcf_3y), "revenue_ttm": r(rev),
            "norm_fcf_margin": r(margin, 4), "fcf_cagr_5y": r(cagr, 4) if cagr else None,
        })
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--export-universe", metavar="CSV")
    ap.add_argument("--universe", default="universe.csv")
    ap.add_argument("-o", "--out", default="universe.json")
    args = ap.parse_args()
    if args.export_universe:
        export_universe_csv(args.export_universe)
        return
    names, source = (fetch_live(args.universe), "live") if args.fetch else (gen_demo(), "demo")
    payload = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": source, "unit": "억원", "terminal_g_default": DEFAULT_TERMINAL_G,
        },
        "sector_wacc": SECTOR_WACC, "names": names,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    ok = cheap = rich = neg = 0
    for nm in names:
        g, st = solve_implied_growth(
            nm["ev"], nm["fcf_ttm"],
            SECTOR_WACC.get(nm["sector"], 0.095), DEFAULT_TERMINAL_G
        )
        if st == "ok":
            ok += 1
            anchor = nm["fcf_cagr_5y"] or 0.03
            if g - anchor < -0.03:
                cheap += 1
            elif g - anchor > 0.03:
                rich += 1
        elif st == "fcf<=0":
            neg += 1
    print(str(args.out) + ": " + str(len(names)) + "종목 source=" + source)
    print("  수렴 " + str(ok) + " | 롱 " + str(cheap) + " | 숏 " + str(rich) + " | 음수FCF " + str(neg))


if __name__ == "__main__":
    main()
