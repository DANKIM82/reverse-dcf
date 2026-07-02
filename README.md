# 역DCF 유니버스 스크리너 (Reverse-DCF Screener)

한국 상장사 유니버스에 대해 **역DCF(implied growth)**를 계산하고, 시장이 EV에
반영 중인 내재 성장률을 과거 FCF 성장(앵커)과 비교해 **저평가/고평가 후보를
스크리닝**하는 단일 파일 도구입니다.

- **`screener.html`** — 브라우저에서 바로 열리는 스크리너(계산·필터·태깅·CSV/워크스페이스 저장). 서버·빌드 불필요
- **`pipeline.py`** — `universe.json` 데이터 생성기(데모 or pykrx+OpenDART 실데이터)
- **`universe.json`** — 스크리너가 읽는 데이터 (스키마는 `pipeline.py` 상단 docstring 참고)

## 빠른 시작 (데모 데이터)

```bash
python pipeline.py --demo -o universe.json   # 가상 수치로 즉시 체험
```

`screener.html`을 브라우저로 열면 됩니다. (같은 폴더의 `universe.json`을 자동 로드,
없으면 파일에 내장된 데모 데이터로 폴백)

## 실데이터로 업데이트

`pykrx`(시가총액) + [OpenDART](https://opendart.fss.or.kr)(재무제표)로 실숫자를 수집합니다.

```bash
pip install -r requirements.txt

# DART 인증키 발급: https://opendart.fss.or.kr  → 인증키 신청 (무료)
export DART_API_KEY=발급키              # Windows PowerShell: $env:DART_API_KEY="발급키"

python pipeline.py --export-universe universe.csv          # 최초 1회: 종목목록 생성
python pipeline.py --fetch --universe universe.csv -o universe.json
```

수집이 끝나면 `screener.html`을 열 때 갱신된 `universe.json`이 자동으로 반영됩니다
(상단 배지가 "실데이터 · 생성시각"으로 표시). 종목을 바꾸려면 `universe.csv`를
직접 편집한 뒤 `--fetch`를 다시 실행하세요.

### 데이터 로딩 방식

- **HTTP 환경**(GitHub Pages, 로컬 서버): 페이지가 `./universe.json`을 자동 fetch
- **오프라인**(`file://` 더블클릭): fetch가 막히므로 HTML에 내장된 데이터로 폴백.
  최신 데이터를 오프라인에서 쓰려면 스크리너 상단 **"데이터 JSON 불러오기"** 버튼으로
  `universe.json`을 직접 로드하면 됩니다.

## GitHub Pages 배포

리포 Settings → Pages에서 소스 브랜치를 지정하면 `screener.html`이 그대로 서빙되고,
같은 경로의 `universe.json`을 자동으로 읽습니다. 데이터 갱신은 `--fetch` 후
`universe.json`을 커밋/푸시하는 것만으로 끝납니다.

## 방법론 주의사항

- **EV = 시가총액 + 순차입금**. FCF는 FCFF 근사(`CFO − CAPEX`)
- **지주사**(`holdco=1`)는 연결 FCF 역DCF가 부적합하여 신호 산출에서 제외(NAV 접근 권장)
- **금융/지주**는 차입금 성격이 특수하여 실데이터 수집 시 `net_debt=0`으로 처리(EV≈시총)
- **TTM은 연간 근사**: 최근 사업보고서(연간) 수치를 TTM 대용으로 사용
- DART 계정과목은 회사별 표기차가 있어 **키워드 매칭으로 근사**하며 **연결(CFS) 우선**.
  핵심 종목은 수치 스팟체크 권장

> 투자 판단의 참고용 도구이며 특정 종목의 매수/매도 권유가 아닙니다.
