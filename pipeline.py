#!/usr/bin/env python3
"""
역DCF 유니버스 스크리너 데이터 파이프라인

사용법
  데모 데이터 생성 (스크리너 즉시 사용):
      python pipeline.py --demo -o universe.json

  실데이터 수집 (로컬 실행, pykrx + OpenDART 필요):
      pip install -r requirements.txt   # pykrx, opendartreader, pandas
      export DART_API_KEY=발급키         # opendart.fss.or.kr 무료 발급
      python pipeline.py --export-universe universe.csv  # 최초 1회: 종목목록 생성
      python pipeline.py --fetch --universe universe.csv -o universe.json

  universe.csv 형식 (헤더 필수. --export-universe로 자동 생성 가능):
      code,name,sector,holdco
      005930,삼성전자,반도체,0
      034730,SK,지주,1

  출력 JSON 스키마 (스크리너 HTML이 읽는 형식):
  {
    "meta": {"generated": iso8601, "source": "demo|live", "unit": "억원"},
    "sector_wacc": {"반도체": 0.105, ...},
    "names": [{
      "code", "name", "sector", "holdco",
      "mktcap", "net_debt", "ev",            # 억원
      "fcf_ttm", "fcf_3y_avg",              # FCFF 근사 = CFO - CAPEX
      "revenue_ttm", "norm_fcf_margin",     # 마진 기반 역DCF용
      "fcf_cagr_5y"                         # 갭 계산 기본 앵커
    }]
  }

  주의
  - EV = 시가총액 + 순차입금. 지주사는 holdco=1로 표기하면 스크리너에서
    신호 산출 대상에서 제외됨 (연결 FCF 역DCF 부적합, NAV 접근 권장).
  - DART 계정과목 매칭은 회사별 표기 차이가 있어 키워드 방식으로 근사함.
    핵심 종목은 수치 스팟체크 권장.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

# ----------------------------------------------------------------------
# 섹터 기본 WACC (스크리너에서 전역 오버라이드 가능)
# ----------------------------------------------------------------------
SECTOR_WACC = {
    "반도체": 0.105, "IT부품": 0.105, "2차전지": 0.11, "인터넷": 0.105,
    "게임": 0.11, "자동차": 0.095, "조선": 0.10, "방산": 0.095,
    "철강화학": 0.095, "바이오": 0.115, "음식료": 0.08, "화장품": 0.095,
    "금융": 0.09, "지주": 0.09, "유틸통신": 0.075, "엔터": 0.11,
    "유통소비": 0.085, "건설기계": 0.10, "운송": 0.10, "기타": 0.095,
}
DEFAULT_TERMINAL_G = 0.01

# ======================================================================
# 역DCF 코어 (스크리너 JS와 동일 로직, 검증/배치 계산용)
# ======================================================================
def solve_implied_growth(ev, fcf0, wacc, tg, n=5, lo=-0.5, hi=1.0):
    """단조함수 이분법. 반환 (g, status)."""
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

# ======================================================================
# pykrx 시총 조회 - 최근 영업일 탐색 (robust)
# ======================================================================
def _get_mktcap_robust(stock, code):
    """
    오늘 기준으로 최대 10 영업일 이전까지 역순으로 탐색하여
    시가총액을 가져옴. GitHub Actions(UTC) 환경 + 장중/휴장 대응.
    """
    # KST 기준 오늘 (UTC+9)
    kst_today = datetime.utcnow() + timedelta(hours=9)
    
    for days_back in range(0, 14):
        target = kst_today - timedelta(days=days_back)
        # 주말 건너뜀
        if target.weekday() >= 5:
            continue
        date_str = target.strftime("%Y%m%d")
        try:
            df = stock.get_market_cap(date_str, date_str, code)
            if df is not None and not df.empty and len(df) > 0:
                val = float(df["시가총액"].iloc[-1])
                if val > 0:
                    return val / 1e8, date_str
        except Exception:
            pass
    return None, None

# ======================================================================
# 실데이터 수집
# ======================================================================
def _import_opendart():
    """opendartreader 모듈을 임포트. 패키지명 대소문자 변형 모두 시도."""
    for mod_name in ("opendartreader", "OpenDartReader", "open_dart_reader"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            return mod
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "opendartreader 모듈을 찾을 수 없습니다. "
        "'pip install opendartreader' 로 설치하세요."
    )


def fetch_live(universe_csv):
    """pykrx 시총 + OpenDART 재무. DART_API_KEY 환경변수 필요."""
    import pandas as pd
    from pykrx import stock

    _odr = _import_opendart()

    api_key = os.environ.get("DART_API_KEY")
    if not api_key:
        sys.exit("DART_API_KEY 환경변수가 필요합니다.")
    _Client = _odr if callable(_odr) else _odr.OpenDartReader
    dart = _Client(api_key)

    uni = pd.read_csv(universe_csv, dtype={"code": str}, encoding="utf-8-sig")
    names = []

    for _, row in uni.iterrows():
        code = row["code"].zfill(6)
        try:
            # ---- 시가총액 (억원) - robust 날짜 탐색 ----
            mktcap, used_date = _get_mktcap_robust(stock, code)
            if mktcap is None:
                print(f"  {code} {row['name']} 실패: 시총 조회 불가 (최근 14일)")
                continue
            print(f"  {code} {row['name']} 시총기준일={used_date}")

            # ---- DART 연간 재무 5개년 + TTM 근사 ----
            fin = collect_dart_financials(dart, code)

            sector = row["sector"]
            net_debt = 0.0 if sector in ("금융", "지주") else fin["net_debt"]

            names.append({
                "code": code, "name": row["name"], "sector": sector,
                "holdco": bool(int(row.get("holdco", 0))),
                "mktcap": round(mktcap, 1),
                "net_debt": round(net_debt, 1),
                "ev": round(mktcap + net_debt, 1),
                "fcf_ttm": fin["fcf_ttm"],
                "fcf_3y_avg": fin["fcf_3y_avg"],
                "revenue_ttm": fin["revenue_ttm"],
                "norm_fcf_margin": fin["norm_fcf_margin"],
                "fcf_cagr_5y": fin["fcf_cagr_5y"],
            })
            time.sleep(0.3)  # KRX + DART rate limit
        except Exception as e:
            print(f"  {code} {row['name']} 실패: {e}")
    return names

def _match_amount(df, keywords, sj_div=None, fs_div="CFS"):
    """계정명 키워드 매칭으로 당기 금액 추출 (억원). 실패 시 None."""
    d = df
    if fs_div is not None and "fs_div" in d.columns:
        cfs = d[d["fs_div"] == fs_div]
        d = cfs if not cfs.empty else d
    if sj_div is not None:
        d = d[d["sj_div"] == sj_div]
    for kw in keywords:
        hit = d[d["account_nm"].str.replace(" ", "").str.contains(kw, na=False)]
        if not hit.empty:
            v = str(hit.iloc[0]["thstrm_amount"]).replace(",", "")
            try:
                return float(v) / 1e8
            except ValueError:
                continue
    return None

def collect_dart_financials(dart, code, years=6):
    """연간 CFO/CAPEX/매출/순차입금 시계열 -> FCF 지표 산출."""
    this_year = datetime.today().year
    fcf_series, rev_series = [], []
    net_debt = 0.0

    for y in range(this_year - years, this_year):
        try:
            df = dart.finstate_all(code, y, reprt_code="11011")
            if df is None or df.empty:
                continue
            cfo = _match_amount(df, ["영업활동현금흐름", "영업활동으로인한현금흐름"], "CF")
            capex_t = _match_amount(df, ["유형자산의취득"], "CF") or 0.0
            capex_i = _match_amount(df, ["무형자산의취득"], "CF") or 0.0
            rev = _match_amount(df, ["매출액", "영업수익", "수익\\(매출액\\)"], "CIS") \
                or _match_amount(df, ["매출액", "영업수익"], "IS")
            if cfo is not None:
                fcf_series.append((y, cfo - abs(capex_t) - abs(capex_i)))
            if rev is not None:
                rev_series.append((y, rev))
            if y == this_year - 1:
                borrow_s = _match_amount(df, ["단기차입금"], "BS") or 0.0
                borrow_l = _match_amount(df, ["장기차입금", "사채"], "BS") or 0.0
                cash     = _match_amount(df, ["현금및현금성자산"], "BS") or 0.0
                stfin    = _match_amount(df, ["단기금융상품"], "BS") or 0.0
                net_debt = borrow_s + borrow_l - cash - stfin
        except Exception:
            continue

    fcf_vals = [v for _, v in fcf_series]
    fcf_ttm  = fcf_vals[-1] if fcf_vals else None
    fcf_3y   = float(np.mean(fcf_vals[-3:])) if len(fcf_vals) >= 3 else fcf_ttm
    rev_ttm  = rev_series[-1][1] if rev_series else None

    cagr = None
    if len(fcf_vals) >= 5 and fcf_vals[0] > 0 and fcf_vals[-1] > 0:
        cagr = (fcf_vals[-1] / fcf_vals[0]) ** (1 / (len(fcf_vals) - 1)) - 1

    margin = None
    if rev_ttm and fcf_vals:
        pos = [v for v in fcf_vals if v > 0]
        if pos:
            margin = min(max(np.median(pos) / rev_ttm, 0.01), 0.35)

    r = lambda x, d=1: None if x is None else round(float(x), d)
    return {
        "fcf_ttm": r(fcf_ttm), "fcf_3y_avg": r(fcf_3y),
        "revenue_ttm": r(rev_ttm), "norm_fcf_margin": r(margin, 4),
        "fcf_cagr_5y": r(cagr, 4), "net_debt": r(net_debt) or 0.0,
    }

# ======================================================================
# 데모 유니버스 (실제 종목명, 수치는 규모감만 맞춘 가상값)
# ======================================================================
DEMO_UNIVERSE = [
    ("005930", "삼성전자",   "반도체",   0, 400, 18), ("000660", "SK하이닉스",  "반도체",   0, 190, 14),
    ("042700", "한미반도체",  "반도체",   0,  12, 38), ("058470", "리노공업",    "반도체",   0,   4, 26),
    ("403870", "HPSP",      "반도체",   0,   3, 30), ("240810", "원익IPS",     "반도체",   0,   2, 22),
    ("011070", "LG이노텍",   "IT부품",   0,   5,  9), ("009150", "삼성전기",    "IT부품",   0,  12, 15),
    ("007660", "이수페타시스","IT부품",   0,   3, 34),
    ("373220", "LG에너지솔루션","2차전지", 0,  85, -1), ("006400", "삼성SDI",    "2차전지",  0,  25, -1),
    ("247540", "에코프로비엠","2차전지",  0,  12, -1), ("003670", "포스코퓨처엠","2차전지",  0,  10, 45),
    ("035420", "NAVER",     "인터넷",   0,  32, 17), ("035720", "카카오",      "인터넷",   0,  18, 24),
    ("259960", "크래프톤",   "게임",     0,  16, 13), ("036570", "엔씨소프트",  "게임",     0,   4, 12),
    ("251270", "넷마블",     "게임",     0,   4, 28),
    ("005380", "현대차",     "자동차",   0,  50,  7), ("000270", "기아",        "자동차",   0,  40,  6),
    ("012330", "현대모비스",  "자동차",   0,  24,  8), ("018880", "한온시스템",  "자동차",   0,   4, 16),
    ("161390", "한국타이어앤테크놀로지","자동차",0,   6,  7),
    ("009540", "HD한국조선해양","조선",   0,  15, 20), ("010140", "삼성중공업",  "조선",     0,  11, 25),
    ("042660", "한화오션",   "조선",     0,  10, -1), ("329180", "HD현대중공업","조선",     0,  20, 28),
    ("012450", "한화에어로스페이스","방산",0,  32, 30), ("079550", "LIG넥스원",  "방산",     0,   8, 27),
    ("064350", "현대로템",   "방산",     0,  12, 26),
    ("005490", "POSCO홀딩스","철강화학", 0,  25,  9), ("010130", "고려아연",    "철강화학", 0,  16, 15),
    ("011170", "롯데케미칼", "철강화학", 0,   3, -1), ("011780", "금호석유",    "철강화학", 0,   4,  8),
    ("051910", "LG화학",    "철강화학", 0,  22, 40),
    ("207940", "삼성바이오로직스","바이오",0,  70, 42), ("068270", "셀트리온",   "바이오",   0,  40, 35),
    ("000100", "유한양행",   "바이오",   0,   9, 35), ("196170", "알테오젠",    "바이오",   0,  18, 55),
    ("097950", "CJ제일제당", "음식료",   0,   5,  8), ("271560", "오리온",      "음식료",   0,   4,  9),
    ("004370", "농심",      "음식료",   0,   2,  9), ("033780", "KT&G",       "음식료",   0,  13, 10),
    ("090430", "아모레퍼시픽","화장품",  0,   8, 22), ("051900", "LG생활건강",  "화장품",   0,   5, 12),
    ("192820", "코스맥스",   "화장품",   0,   2, 13), ("161890", "한국콜마",    "화장품",   0,   2, 12),
    ("257720", "실리콘투",   "화장품",   0,   2, 24),
    ("105560", "KB금융",    "금융",     0,  35,  6), ("055550", "신한지주",    "금융",     0,  27,  6),
    ("086790", "하나금융지주","금융",    0,  18,  5), ("316140", "우리금융지주","금융",     0,  12,  5),
    ("138040", "메리츠금융지주","금융",  0,  22,  8), ("000810", "삼성화재",    "금융",     0,  17,  8),
    ("005830", "DB손해보험", "금융",     0,   8,  6), ("323410", "카카오뱅크",  "금융",     0,  10, 15),
    ("034730", "SK",        "지주",     1,  13,  5), ("003550", "LG",         "지주",     1,  12,  7),
    ("028260", "삼성물산",   "지주",     1,  22, 11), ("000880", "한화",        "지주",     1,   3,  4),
    ("006260", "LS",        "지주",     1,   4,  7), ("001040", "CJ",         "지주",     1,   3,  5),
    ("000150", "두산",      "지주",     1,   6, 12),
    ("015760", "한국전력",   "유틸통신", 0,  15, -1), ("030200", "KT",         "유틸통신", 0,  12,  6),
    ("017670", "SK텔레콤",  "유틸통신", 0,  12,  7), ("032640", "LG유플러스",  "유틸통신", 0,   4,  5),
    ("352820", "하이브",    "엔터",     0,  10, 30), ("035900", "JYP Ent.",   "엔터",     0,   2, 14),
    ("041510", "에스엠",    "엔터",     0,   2, 15), ("122870", "와이지엔터테인먼트","엔터",0,   1, 18),
    ("021240", "코웨이",    "유통소비", 0,   6,  9), ("139480", "이마트",      "유통소비", 0,   2, -1),
    ("282330", "BGF리테일", "유통소비", 0,   2,  8), ("007070", "GS리테일",    "유통소비", 0,   2,  7),
    ("008770", "호텔신라",   "유통소비", 0,   2, 20),
    ("000720", "현대건설",   "건설기계", 0,   4,  7), ("034020", "두산에너빌리티","건설기계",0,  15, 48),
    ("241560", "두산밥캣",  "건설기계", 0,   5,  6), ("267260", "HD현대일렉트릭","건설기계",0,  14, 32),
    ("010120", "LS일렉트릭","건설기계", 0,   6, 25), ("298040", "효성중공업",  "건설기계", 0,   5, 28),
    ("003490", "대한항공",   "운송",     0,   9,  7), ("011200", "HMM",        "운송",     0,  16,  5),
    ("086280", "현대글로비스","운송",    0,  10,  8),
]

def export_universe_csv(path):
    """DEMO_UNIVERSE의 종목 목록을 --fetch용 universe.csv로 내보냄."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "sector", "holdco"])
        for code, name, sector, holdco, *_ in DEMO_UNIVERSE:
            w.writerow([code, name, sector, holdco])
    print(f"{path} 저장: {len(DEMO_UNIVERSE)}종목")

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
            fcf_3y  = fcf_ttm * rng.uniform(0.3, 1.2)
            cagr    = None
        else:
            fcf_ttm = ev / (evfcf_hint * rng.uniform(0.85, 1.2))
            fcf_3y  = fcf_ttm * rng.uniform(0.8, 1.15)
            cagr    = float(np.clip(rng.normal(0.05, 0.09), -0.20, 0.35))

        rev    = mktcap * rng.uniform(0.4, 1.6)
        margin = float(np.clip(abs(fcf_3y) / rev * rng.uniform(0.9, 1.3), 0.02, 0.25))

        r = lambda x, d=1: None if x is None else round(float(x), d)
        names.append({
            "code": code, "name": name, "sector": sector, "holdco": bool(holdco),
            "mktcap": r(mktcap), "net_debt": r(net_debt), "ev": r(ev),
            "fcf_ttm": r(fcf_ttm), "fcf_3y_avg": r(fcf_3y),
            "revenue_ttm": r(rev), "norm_fcf_margin": r(margin, 4),
            "fcf_cagr_5y": r(cagr, 4) if cagr is not None else None,
        })
    return names

# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo",   action="store_true", help="데모 데이터 생성")
    ap.add_argument("--fetch",  action="store_true", help="pykrx+DART 실데이터 수집")
    ap.add_argument("--export-universe", metavar="CSV",
                    help="DEMO_UNIVERSE 종목을 --fetch용 CSV로 내보내고 종료")
    ap.add_argument("--universe", default="universe.csv")
    ap.add_argument("-o", "--out", default="universe.json")
    args = ap.parse_args()

    if args.export_universe:
        export_universe_csv(args.export_universe)
        return

    if args.fetch:
        names, source = fetch_live(args.universe), "live"
    else:
        names, source = gen_demo(), "demo"

    payload = {
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "source": source, "unit": "억원",
                 "terminal_g_default": DEFAULT_TERMINAL_G},
        "sector_wacc": SECTOR_WACC,
        "names": names,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    ok = cheap = rich = neg = 0
    for nm in names:
        g, st = solve_implied_growth(nm["ev"], nm["fcf_ttm"],
                                     SECTOR_WACC.get(nm["sector"], 0.095),
                                     DEFAULT_TERMINAL_G)
        if st == "ok":
            ok += 1
            anchor = nm["fcf_cagr_5y"] or 0.03
            if   g - anchor < -0.03: cheap += 1
            elif g - anchor >  0.03: rich  += 1
        elif st == "fcf<=0":
            neg += 1
    print(f"{args.out} 저장: {len(names)}종목 (source={source})")
    print(f"  해 수렴 {ok} | 저평가 후보 {cheap} | 고평가 후보 {rich} | 음수FCF {neg}")

if __name__ == "__main__":
    main()
