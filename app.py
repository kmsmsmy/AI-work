"""SSE-AI Lab — 2차 중간발표 전용 시연 화면 (계산은 v29.1 그대로 유지).

기존 Colab에서 직접 저장한 SSE_AI_Model_v1.joblib이 이 파일과 같은 폴더에 필요합니다.
OBELiX 학습 데이터는 '소재 데이터베이스 불러오기'를 눌렀을 때 공식 저장소에서
다운로드되어 OBELiX_train.csv로 보관됩니다. 테스트 데이터는 불러오지 않습니다.

데이터 출처: NRC-Mila/OBELiX, CC BY 4.0
https://github.com/NRC-Mila/OBELiX
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, r2_score

APP_VERSION = "v29.3"
APP_FILE_TAG = APP_VERSION.replace(".", "_")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "SSE_AI_Model_v1.joblib"
MODEL_V2_PATH = BASE_DIR / "SSE_AI_Model_v2.joblib"
DATABASE_PATH = BASE_DIR / "OBELiX_train.csv"
TRAIN_REFERENCE_PATH = BASE_DIR / "SSE_train_reference.csv"
VALID_REFERENCE_PATH = BASE_DIR / "SSE_valid_reference.csv"
DATABASE_URL = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv"
DATASET_PAGE = "https://github.com/NRC-Mila/OBELiX"
TARGET_COL = "Ionic conductivity (S cm-1)"
FORMULA_COL = "Reduced Composition"

st.set_page_config(page_title="SSE-AI Lab | 2차 중간발표", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")
st.title("🔋 SSE-AI Lab")
st.caption("고체전해질 이온전도도 예측 · 조성 탐색 · 문헌 분석 · 모델 검증")
st.caption("연구용 참고 도구입니다. AI 예측값은 실험 측정값이 아니며 정확도가 보장되지 않습니다.")



# ─────────────────────────────────────────────────────────────────────────────
# 입력 편의성 패치: 아래첨자·전각 숫자·대문자/소문자 보정
# (원본 학습 데이터/모델의 조성값은 변경하지 않습니다.)
# ─────────────────────────────────────────────────────────────────────────────
import re
import unicodedata
from functools import lru_cache

# 공식 원소 기호. 대소문자 보정이 모델 특징에만 의존하지 않도록 전체 원소를 사용합니다.
_PERIODIC_SYMBOLS = frozenset('''H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'''.split())
# 조성 기반 전고체전해질에서 많이 쓰이는 원소의 가중치. 소문자만 입력했을 때
# 'po4'를 Po4(폴로늄)로 잘못 읽지 않고 P+O4(인+산소)로 읽는 데 사용합니다.
_COMMON_SSE_ELEMENTS = frozenset('Li P S O Cl Br I La Zr Y Ge Si B N Na K Ti Al Nb Ta Sn In F Mg Ca Cu Fe Co Mn Zn H C Se As W Ba Sr Sc Sb Ga'.split())
_RARE_RADIOACTIVE = frozenset('Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Pm Tc'.split())
_SUBSCRIPT_DIGITS = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def _plain_formula_characters(raw: str) -> str:
    """입력 검색에도 사용: 유니코드 숫자 표기를 일반 숫자로 변환."""
    if not isinstance(raw, str):
        raise ValueError('화학식은 글자로 입력해주세요.')
    result = unicodedata.normalize('NFKC', raw).translate(_SUBSCRIPT_DIGITS)
    result = re.sub(r'\s+', '', result)
    return result


@lru_cache(maxsize=256)
def _repair_element_sequence(run: str) -> str:
    """원소 기호의 대/소문자가 섞인 알파벳 덩어리를 복원합니다.

    표준 원소 대문자 표기가 이미 정확하면 반드시 원본을 보존합니다.
    소문자/대문자만 쓴 경우에는 실제 주기율표 기호로 분할합니다.
    CO/Co, SI/Si 등 대소문자 자체가 원소 뜻을 바꾸는 경우는 본질적으로
    모호할 수 있으므로 화면에서 최종 인식된 화학식을 확인하도록 합니다.
    """
    if not run.isascii() or not run.isalpha():
        raise ValueError('화학식에는 영문 원소 기호를 입력해주세요.')
    conventional = re.findall(r'[A-Z][a-z]?', run)
    if ''.join(conventional) == run and all(s in _PERIODIC_SYMBOLS for s in conventional):
        return run

    low = run.lower()

    @lru_cache(maxsize=None)
    def search(i):
        if i == len(low):
            return (0.0, ())
        best = None
        for size in (1, 2):
            sym = low[i:i + size]
            if len(sym) != size:
                continue
            symbol = sym[0].upper() + sym[1:]
            if symbol not in _PERIODIC_SYMBOLS:
                continue
            rest = search(i + size)
            if rest is None:
                continue
            # 원소 1개를 너무 많이 쪼개지 않도록 약한 벌점을 부여.
            # 희귀/방사성 원소는 소문자 자동 보정에서만 매우 보수적으로 취급.
            weight = (5.0 if symbol in _COMMON_SSE_ELEMENTS else 2.0)
            if symbol in _RARE_RADIOACTIVE:
                weight -= 15.0
            # 화학식에서 Nb, Sn, Si, In 같은 두 글자 원소를
            # 'N+B', 'S+N', 'S+I', 'I+N'으로 쪼개는 오류를 방지합니다.
            score = rest[0] + weight - 3.5 + (2.6 if size == 2 else 0.0)
            # 소문자/대문자 섞어 쓰기는 기존 대문자 경계를 힌트로 사용.
            if i + 1 < len(run) and size == 2 and run[i+1].isupper() and run[i].isupper():
                score -= 1.0
            if best is None or score > best[0]:
                best = (score, (symbol,) + rest[1])
        return best

    answer = search(0)
    if answer is None:
        raise ValueError(f'원소 기호를 인식하지 못했습니다: {run}. 예: Li6PS5Cl')
    return ''.join(answer[1])


def normalize_user_formula(raw: str) -> str:
    """아래첨자·대소문자가 섞인 사용자 입력을 pymatgen용 표준 문자열로 보정.

    숫자, 괄호, 소수점 등 화학양론 정보는 변환하지 않습니다.
    반드시 이후 pymatgen Composition으로 유효성을 검증합니다.
    """
    formula = _plain_formula_characters(raw)
    if not formula:
        raise ValueError('화학식을 입력해주세요. 예: Li6PS5Cl')
    if len(formula) > 120:
        raise ValueError('화학식은 120자 이내로 입력해주세요.')
    if not re.fullmatch(r'[A-Za-z0-9.()\[\]{}+\-]+', formula):
        raise ValueError('영문 원소 기호와 숫자·소수점·괄호만 사용해주세요. 예: Li₆PS₅Cl')
    return re.sub(r'[A-Za-z]+', lambda m: _repair_element_sequence(m.group()), formula)



# ─────────────────────────────────────────────────────────────────────────────
# v29 화면 안정화: 오류 안내, 좁은 화면에 적합한 요약 표
# ─────────────────────────────────────────────────────────────────────────────
def v29_formula_error_hint(formula: str, exc: Exception) -> str:
    """입력 오류에만 덧붙이는 도움말. 모델 계산 결과/입력값을 수정하지 않음."""
    s = str(exc)
    if not str(formula).strip():
        return "화학식을 먼저 입력하세요. 예: Li6PS5Cl"
    if "학습하지 않은 원소" in s or "사용하지 않은 원소" in s:
        return "현재 모델의 지원 원소를 확인해 주세요. 형식이 옳아도 학습에 없는 원소는 예측할 수 없습니다."
    if "화학식" in s or "원소" in s or "parse" in s.lower():
        return "일반 숫자·아래첨자·대소문자를 지원합니다. 예: Li6PS5Cl, Li₆PS₅Cl, li6ps5cl. CO/Co처럼 원소 뜻이 모호한 표기는 표준 대소문자로 입력해 주세요."
    return "모델 파일과 학습 때 사용한 라이브러리 버전, 입력 데이터를 확인해 주세요."


def v29_show_table(full: pd.DataFrame, compact: list[str], *, key: str,
                   height: int = 430) -> None:
    """긴 표는 핵심 열만 먼저 보여주고 전체 열/원본 다운로드는 그대로 제공."""
    small_cols = [c for c in compact if c in full.columns]
    if not small_cols or len(small_cols) == len(full.columns):
        st.dataframe(full, use_container_width=True, hide_index=True, height=height)
        return
    mode = st.radio("표 열 표시", ["핵심 열만", "전체 열"], key=key,
                    horizontal=True, label_visibility="collapsed")
    shown = full[small_cols] if mode == "핵심 열만" else full
    st.dataframe(shown, use_container_width=True, hide_index=True, height=height)
    if mode == "핵심 열만":
        st.caption("핵심 열만 표시 중입니다. '전체 열'을 선택하거나 아래 CSV를 다운로드하면 모든 수치를 볼 수 있습니다.")


@st.cache_resource
def load_model():
    # 사용자 본인이 학습·저장한 신뢰할 수 있는 joblib 파일만 불러오세요.
    saved = joblib.load(MODEL_PATH)
    if not isinstance(saved, dict) or not {"model", "feature_columns"} <= set(saved):
        raise ValueError("모델 저장 형식이 다릅니다. Colab에서 저장한 파일을 확인하세요.")
    columns = list(saved["feature_columns"])
    if not columns:
        raise ValueError("모델의 입력 특징 목록이 비어 있습니다.")
    return saved["model"], columns


@st.cache_data(show_spinner=False)
def load_database():
    """공식 train.csv만 읽고 로컬에 캐시한다. 테스트셋은 불러오지 않는다."""
    if DATABASE_PATH.is_file():
        raw = pd.read_csv(DATABASE_PATH, dtype=str).fillna("")
    else:
        req = Request(DATABASE_URL, headers={"User-Agent": "SSE-AI-Lab/2.0"})
        with urlopen(req, timeout=15) as res:
            content = res.read()
        raw = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
        try:
            DATABASE_PATH.write_bytes(content)
        except OSError:
            # 배포 환경이 읽기 전용일 수 있어도 이번 세션에서는 표시 가능
            pass

    required = {FORMULA_COL, TARGET_COL, "Family", "DOI"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"데이터베이스 열이 누락됐습니다: {', '.join(sorted(missing))}")

    raw["숫자형 측정값"] = pd.to_numeric(raw[TARGET_COL], errors="coerce")
    raw["양의 수치 여부"] = raw["숫자형 측정값"].gt(0)
    raw["실험값 표시"] = raw[TARGET_COL].astype(str)
    raw["논문 링크"] = raw["DOI"].apply(
        lambda x: f"https://doi.org/{x.strip()}" if str(x).strip().startswith("10.") else ""
    )
    return raw


if not MODEL_PATH.is_file():
    st.error("AI 모델 파일을 찾을 수 없습니다.")
    st.write("`SSE_AI_Model_v1.joblib` 파일을 `app.py`와 **같은 폴더**에 넣어주세요.")
    st.stop()

try:
    model, feature_columns = load_model()
except Exception as exc:
    st.error("학습된 AI 모델을 불러오지 못했습니다.")
    st.code(str(exc))
    st.caption("Colab과 PC의 scikit-learn 버전 차이 때문에 문제가 생길 수도 있습니다.")
    st.stop()


@st.cache_resource
def load_model_v2():
    """사용자가 Colab에서 직접 학습한 59개 특징 모델을 로드한다."""
    saved = joblib.load(MODEL_V2_PATH)
    if not isinstance(saved, dict) or not {"model", "feature_columns"} <= set(saved):
        raise ValueError("v2 저장 파일에 model 또는 feature_columns 항목이 없습니다.")
    columns = list(saved["feature_columns"])
    if len(columns) != 59:
        raise ValueError(f"v2 모델 특징 수가 59개가 아닙니다: {len(columns)}개")
    if saved.get("feature_type") not in (None, "composition_plus_7"):
        raise ValueError("v2 모델 특징 유형 정보가 예상과 다릅니다.")
    return saved["model"], columns


def material_features_7(formula: str) -> dict:
    """Colab에서 v2 학습에 사용했던 material_features 계산을 동일하게 재현."""
    comp = Composition(normalize_user_formula(str(formula)))
    elements = list(comp.items())
    total = comp.num_atoms
    weights = np.array([amount / total for el, amount in elements])

    atomic_numbers = np.array([float(el.Z) for el, amount in elements])
    atomic_masses = np.array([float(el.atomic_mass) for el, amount in elements])
    electronegativities = np.array([
        float(el.X) if el.X is not None else np.nan
        for el, amount in elements
    ])
    atomic_radii = np.array([
        float(el.atomic_radius) if el.atomic_radius is not None else np.nan
        for el, amount in elements
    ])

    def weighted_mean(values):
        mask = np.isfinite(values)
        if not mask.any():
            return np.nan
        return np.average(values[mask], weights=weights[mask])

    def value_range(values):
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan
        return valid.max() - valid.min()

    return {
        "number_of_elements": len(elements),
        "mean_atomic_number": weighted_mean(atomic_numbers),
        "mean_atomic_mass": weighted_mean(atomic_masses),
        "mean_electronegativity": weighted_mean(electronegativities),
        "range_electronegativity": value_range(electronegativities),
        "mean_atomic_radius": weighted_mean(atomic_radii),
        "range_atomic_radius": value_range(atomic_radii),
    }


def make_input_v2(formula: str, v2_columns: list[str]) -> pd.DataFrame:
    """52개 조성비 + 7개 물성 특징, 저장된 순서대로 정렬."""
    base = make_input(formula)
    physical = material_features_7(formula)
    features = {**base.iloc[0].to_dict(), **physical}
    missing = set(v2_columns) - set(features)
    if missing:
        raise ValueError(f"v2 입력에 필요한 특징이 누락됐습니다: {sorted(missing)}")
    return pd.DataFrame([features]).reindex(columns=v2_columns)


def make_input(formula: str) -> pd.DataFrame:
    """Colab에서 사용한 것과 같은 '원소별 원자수 비율' 입력을 생성."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("화학식을 입력해주세요. 예: Li6PS5Cl, Li3PS4")
    if len(formula.strip()) > 120:
        raise ValueError("입력한 화학식이 너무 깁니다. 120자 이내로 입력해주세요.")
    try:
        comp = Composition(normalize_user_formula(formula))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError(f"화학식 형식을 확인해주세요: {exc}") from exc
    if not np.isfinite(comp.num_atoms) or comp.num_atoms <= 0:
        raise ValueError("화학식에 양의 원자수가 없습니다.")
    if any(not np.isfinite(float(n)) or float(n) <= 0 for _el, n in comp.items()):
        raise ValueError("각 원소의 원자수는 0보다 큰 숫자여야 합니다.")
    features = {el.symbol: float(n / comp.num_atoms) for el, n in comp.items()}
    unsupported = sorted(set(features) - set(feature_columns))
    if unsupported:
        raise ValueError(f"현재 모델이 학습하지 않은 원소: {', '.join(unsupported)}")
    return pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)


def predict_one(formula: str):
    clean_formula = normalize_user_formula(formula)
    if not clean_formula:
        raise ValueError("화학식을 입력해주세요.")
    one_row = make_input(clean_formula)
    log_value = float(model.predict(one_row)[0])
    conductivity = 10.0 ** log_value
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("예측값이 유효한 양수가 아닙니다.")
    return {
        "formula": Composition(clean_formula).reduced_formula,
        "conductivity_S_per_cm": float(conductivity),
        "log10_conductivity": log_value,
    }


def add_prediction(formula: str):
    result = predict_one(formula)
    # 같은 화학식은 비교 목록에서 최신 결과로 갱신
    st.session_state.results = [
        r for r in st.session_state.results if r.get("formula") != result["formula"]
    ]
    st.session_state.results.append(result)
    return result


if "results" not in st.session_state:
    st.session_state.results = []
if "db_opened" not in st.session_state:
    st.session_state.db_opened = False

# v12: 기존 9개 기능의 본문은 유지하고 메뉴 선택 방식만 사이드바로 변경한다.
# 탭은 가로 폭을 넘기 쉽고 비활성 탭 코드도 매번 실행하므로, 현재 메뉴만 렌더링한다.
# 이번 발표에서 직접 설명하는 기능 + 질문에 대비한 문헌 검색/소재 추천만 노출합니다.
# 실험용 구조 예측/최종 배포 점검/보고서 메뉴는 기존 코드에 보존하되 발표 화면에서는 숨깁니다.
MIDTERM_MENU = {
    "predict": "🔮 AI 이온전도도 예측",
    "simulation": "🧪 조성 변경 시뮬레이션",
    "data_quality": "🧾 문헌 측정값 분석",
    "validation": "📈 AI 모델 성능 검증",
    "structure": "🧬 결정구조 분석",
}
V12_MENU = MIDTERM_MENU

if "midterm_nav" not in st.session_state or st.session_state["midterm_nav"] not in MIDTERM_MENU:
    st.session_state["midterm_nav"] = "predict"

with st.sidebar:
    st.markdown("## 🔋 SSE-AI Lab")
    st.caption(f"2차 중간발표 · 발표 순서대로 구성 · {APP_VERSION}")
    st.divider()
    selected_page = st.radio(
        "발표 화면 선택",
        options=list(MIDTERM_MENU),
        format_func=lambda page: MIDTERM_MENU[page],
        key="midterm_nav",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("2차 중간발표 · 핵심 기능 5개")
    with st.expander("ℹ️ 데이터·모델 상태와 주의사항", expanded=False):
        st.caption("52개 원소 조성 특징 모델 연결")
        st.caption("59개 특징 모델 연결" if MODEL_V2_PATH.is_file() else "⚠️ v2 모델 파일 필요")
        st.caption("원본 측정 CSV 사용 가능" if DATABASE_PATH.is_file() else "원본 CSV는 최초 실행 시 불러옵니다")
        st.caption("5겹 교차검증 결과 사용 가능" if (BASE_DIR / "SSE_v14_cv_summary.csv").is_file() else "⚠️ 교차검증 CSV 필요")
        st.write("가상 치환의 최고 예측값은 실제 합성 가능성이나 실험 최적 조성을 의미하지 않습니다.")
        st.write("학습 조성 유사도 및 모델 간 차이는 예측 정확도 확률이 아닙니다.")


def v12_go_to(page_id: str):
    if page_id in MIDTERM_MENU:
        st.session_state["midterm_nav"] = page_id


if selected_page == "home":
    st.subheader("2차 중간발표 | 구현 기능과 모델 검증")
    st.write(
        "**고체전해질의 화학식을 입력하면 이온전도도를 예측하는 AI 웹 앱**입니다. "
        "이번 발표에서는 지난 발표 이후 구현한 **조성 시뮬레이션과 범용 원소 치환, 반복 측정값 분석, "
        "학습 방식 비교와 교차검증**을 실제 화면으로 보여줍니다."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("유효 실험 측정 기록", "449건")
    c2.metric("서로 다른 조성", "362종")
    c3.metric("반복 측정 조성", "55종")
    c4.metric("검증 방식", "5겹 교차검증")
    st.caption("기존 OBELiX 학습 자료의 데이터 분석 결과입니다. 449건은 새로 확보한 데이터 수가 아닙니다.")

    st.markdown("### 발표 시연 순서")
    tasks = [
        ("predict", "1. 화학식으로 전도도 예측", "Li6PS5Cl을 입력하고 '이온전도도 예측'을 누릅니다. AI 추정치와 문헌 측정값을 구별합니다."),
        ("simulation", "2. Cl/Br 치환 21개 조성 비교", "Br 50%를 고르고 21개 조성 분석 버튼을 누릅니다. 곡선은 모델 예측이며 실험값이 아닙니다."),
        ("general_substitution", "추가. 범용 원소 치환 21개 조성", "Li10Ge(PS6)2를 예시로 치환 원소를 선택해 21개 조성을 예측합니다. 실험·합성 가능성은 보장하지 않습니다."),
        ("data_quality", "3. 같은 조성의 다른 측정값", "원본 데이터 분석 버튼을 누른 뒤 Li6PS5Cl을 검색합니다. 논문별 측정값과 62배 차이를 설명합니다."),
        ("evaluation", "4. 내부 검증과 공식 테스트", "354건 학습·95건 검증과 별도 공식 테스트 109건의 의미를 구분합니다."),
        ("cv_review", "5. A·B·C 학습 방식 교차검증", "조성·논문을 함께 분리한 검증을 골라 MAE·R²를 비교합니다."),
        ("structure", "6. 같은 화학식의 다른 결정구조", "Li7La3Zr2O12를 검색해 같은 조성에서 공간군 2종을 확인합니다."),
    ]
    for idx in range(0,len(tasks),2):
        cols = st.columns(2, gap="large")
        for col, (page_id,title,desc) in zip(cols,tasks[idx:idx+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(desc)
                    st.button("이 화면 열기", key=f"midterm_open_{page_id}",
                              on_click=v12_go_to, args=(page_id,), use_container_width=True)
    with st.expander("지난 발표에서 개발한 기능 — 참고 시연"):
        c1,c2=st.columns(2)
        c1.button("문헌 검색 열기", key="midterm_go_database", on_click=v12_go_to, args=("database",),use_container_width=True)
        c2.button("소재 추천 열기", key="midterm_go_recommend", on_click=v12_go_to, args=("recommend",),use_container_width=True)
    st.warning(
        "**이번 발표의 결론:** 데이터 품질과 검증 방식을 개선하고 A·B·C 모델을 비교했지만, "
        "예측 정확도의 일관된 향상은 확인되지 않았습니다. AI는 합성 전 후보 탐색을 돕는 보조 도구입니다."
    )

if selected_page == "predict":
    st.write("화학식을 입력하면 기존 AI 모델이 이온전도도를 예측합니다. 여러 소재를 예측하면 아래에서 비교할 수 있습니다.")
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("1. 화학식 입력 및 AI 예측")
        st.caption("일반 숫자·아래첨자·대문자·소문자 모두 입력할 수 있습니다. 예: `Li₆PS₅Cl`, `li6ps5cl`, `LI6PS5CL`.")
        with st.form("prediction_form"):
            formula = st.text_input(
                "전해질 화학식", "Li6PS5Cl",
                help="아래첨자·일반 숫자·대소문자 혼용 가능. 예: Li₆PS₅Cl, li6ps5cl, LI6PS5CL",
            )
            submitted = st.form_submit_button("🔎 이온전도도 예측", use_container_width=True)
    with right:
        st.subheader("모델 정보")
        st.metric("사용 중인 모델", "Extra Trees")
        st.metric("입력 특징", f"{len(feature_columns)}개")
        st.caption("원소 조성비를 사용하는 v1 모델입니다.")

    if submitted:
        try:
            result = add_prediction(formula)
            st.success(f"{result['formula']} 예측 완료")
            canonical = normalize_user_formula(formula)
            if formula.strip() != canonical:
                st.caption(f"입력 자동 보정: `{formula.strip()}` → `{canonical}`")
            a, b = st.columns(2)
            a.metric("예측 이온전도도 (S/cm)", f"{result['conductivity_S_per_cm']:.4e}")
            b.metric("log₁₀(S/cm)", f"{result['log10_conductivity']:.4f}")
        except Exception as exc:
            st.error(f"예측할 수 없습니다: {exc}")
            st.info(v29_formula_error_hint(formula, exc))

    st.divider()
    st.subheader("2. 소재별 예측 결과 비교")
    if st.session_state.results:
        data = pd.DataFrame(st.session_state.results)
        data = data.drop_duplicates(subset=["formula"], keep="last")
        data = data.sort_values("conductivity_S_per_cm", ascending=False)

        pretty = pd.DataFrame({
            "화학식": data["formula"].to_list(),
            "예측 이온전도도 (S/cm)": data["conductivity_S_per_cm"].map(lambda v: f"{v:.4e}").to_list(),
            "log₁₀(S/cm)": data["log10_conductivity"].map(lambda v: f"{v:.4f}").to_list(),
        })
        st.dataframe(pretty, use_container_width=True, hide_index=True)
        st.caption("수치가 작은 소재도 정확히 구분할 수 있도록 이온전도도를 지수 형태로 표시했습니다.")

        fig = px.scatter(
            data, x="log10_conductivity", y="formula",
            text=data["conductivity_S_per_cm"].map(lambda v: f"{v:.2e}"),
            labels={"formula": "전해질", "log10_conductivity": "이온전도도 (S/cm, 로그 눈금)"},
            title="소재별 AI 예측 이온전도도",
            custom_data=["conductivity_S_per_cm"],
        )
        fig.update_traces(
            mode="markers+text", textposition="middle right",
            marker={"size": 14, "color": "#4c9bef"},
            hovertemplate="%{y}<br>AI 예측값: %{customdata[0]:.4e} S/cm<extra></extra>",
        )
        min_log = float(data["log10_conductivity"].min())
        max_log = float(data["log10_conductivity"].max())
        low, high = int(np.floor(min_log)), int(np.ceil(max_log))
        ticks = list(range(low, high + 1))
        if len(ticks) <= 18:
            fig.update_xaxes(
                tickmode="array", tickvals=ticks,
                ticktext=[f"10<sup>{v}</sup>" for v in ticks],
            )
        fig.update_xaxes(range=[min_log - 0.6, max_log + 1.7], showgrid=True)
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(data["formula"].tolist())))
        fig.update_layout(height=max(310, 65 * len(data) + 140), margin=dict(l=10, r=35, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("오른쪽에 있는 점일수록 예측 이온전도도가 높습니다. 점 옆에는 실제 S/cm 단위의 예측값을 표기했습니다.")

        with st.expander("결과 저장·비교 목록 관리", expanded=False):
            st.download_button(
                "📥 예측 결과 CSV 저장",
                data=data.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_AI_predictions.csv", mime="text/csv",
            )
            if st.button("🗑️ 비교 목록 지우기"):
                st.session_state.results = []
                st.rerun()
    else:
        st.caption("화학식을 예측하면 이곳에 비교표와 그래프가 나타납니다.")

if selected_page == "database":
    st.subheader("OBELiX 고체전해질 소재 데이터베이스")
    st.write("공식 OBELiX **학습 데이터**에서 화학식, 소재 계열, 문헌 이온전도도와 DOI를 검색합니다.")
    st.caption("공식 최종 테스트 데이터는 이 검색 기능에 포함하지 않습니다. 데이터 출처: NRC-Mila/OBELiX (CC BY 4.0).")
    st.link_button("📖 OBELiX 공식 출처", DATASET_PAGE)

    if st.button("📥 소재 데이터베이스 불러오기", use_container_width=True):
        st.session_state.db_opened = True

    if st.session_state.db_opened:
        try:
            with st.spinner("OBELiX 학습 데이터를 읽는 중..."):
                db = load_database()
        except Exception as exc:
            st.error("소재 데이터를 불러오지 못했습니다. 인터넷 연결을 확인해주세요.")
            st.code(str(exc))
            st.write("대안: 아래 공식 CSV 파일을 받아서 `OBELiX_train.csv`라는 이름으로 `app.py` 옆에 놓은 다음 앱을 다시 실행하세요.")
            st.link_button("공식 학습 CSV 열기", DATABASE_URL)
            st.stop()

        x, y = st.columns(2)
        x.metric("학습 데이터 전체", f"{len(db)}건")
        y.metric("양의 수치형 이온전도도", f"{int(db['양의 수치 여부'].sum())}건")

        col_query, col_family = st.columns([2, 1])
        with col_query:
            query = st.text_input("화학식 검색", placeholder="예: Li6PS5Cl 또는 Li3PS4", key="db_search")
        with col_family:
            families = sorted(s for s in db["Family"].unique() if str(s).strip())
            selected_family = st.selectbox("소재 계열", ["전체"] + families)

        exact_only = st.checkbox("정확한 양의 숫자로 표시된 측정값만 보기", value=False)
        filtered = db.copy()
        if query.strip():
            filtered = filtered[filtered[FORMULA_COL].str.contains(_plain_formula_characters(query), case=False, regex=False, na=False)]
        if selected_family != "전체":
            filtered = filtered[filtered["Family"] == selected_family]
        if exact_only:
            filtered = filtered[filtered["양의 수치 여부"]]

        st.write(f"**검색 결과: {len(filtered)}건**")
        if not filtered.empty:
            view = filtered[["ID", FORMULA_COL, "Family", "실험값 표시", "Space group", "논문 링크"]].copy()
            view.columns = ["데이터 ID", "화학식", "소재 계열", "문헌 이온전도도 (S/cm)", "공간군", "논문 DOI"]
            st.dataframe(
                view, use_container_width=True, hide_index=True,
                column_config={"논문 DOI": st.column_config.LinkColumn("논문 DOI", display_text="원문 보기")},
            )
            st.caption("'<1E-10' 등은 검출 한계 또는 범위를 나타내는 표기이며, 정확한 측정값으로 바꾸지 않았습니다.")
            st.caption("같은 화학식이 여러 번 등장할 수 있습니다. 문헌·결정구조·측정 조건이 다를 수 있기 때문입니다.")

            st.markdown("#### 검색된 소재를 직접 AI로 예측해 보기")
            options = filtered.index.tolist()
            selected_idx = st.selectbox(
                "문헌 데이터 선택", options,
                format_func=lambda i: (
                    f"{db.loc[i, FORMULA_COL]} · {db.loc[i, 'Family']} · "
                    f"측정 {db.loc[i, TARGET_COL]} S/cm · ID {db.loc[i, 'ID']}"
                ),
                key="db_record",
            )
            if st.button("🔮 선택한 소재 AI 예측하기"):
                db_formula = str(db.loc[selected_idx, FORMULA_COL])
                try:
                    result = add_prediction(db_formula)
                    st.success(f"{db_formula}의 예측 결과를 비교 목록에 추가했습니다.")
                    c1, c2 = st.columns(2)
                    c1.metric("AI 예측값 (S/cm)", f"{result['conductivity_S_per_cm']:.4e}")
                    measured = db.loc[selected_idx, TARGET_COL]
                    c2.metric("문헌 기록 (S/cm)", str(measured))
                    st.warning("이 문헌 자료는 모델 학습에 사용되었을 수 있습니다. 이 비교는 독립 검증 성능을 뜻하지 않습니다.")
                except Exception as exc:
                    st.error(f"선택한 조성을 예측할 수 없습니다: {exc}")
        else:
            st.caption("검색 조건에 맞는 자료가 없습니다. 화학식이나 소재 계열을 변경해 보세요.")



# ─────────────────────────────────────────────────────────────────────────────
# 세 번째 메뉴: 이미 문헌에 기록된 조성들 중 AI 예측값이 높은 후보 찾기
# ─────────────────────────────────────────────────────────────────────────────
def predict_catalog(database: pd.DataFrame):
    """공식 *학습용* 문헌 DB를 대상으로 예측, 중복 화학식은 하나로 묶는다.

    실험에서 검증된 새로운 소재를 발명하거나 검증하는 기능이 아니라,
    데이터베이스에 이미 존재하는 조성들을 검색·정렬하는 기능이다.
    """
    valid_db = database[database["양의 수치 여부"]].copy()
    vectors, records = [], []
    rejected = 0
    for _, r in valid_db.iterrows():
        formula = str(r[FORMULA_COL]).strip()
        try:
            feat = make_input(formula)
            normalized = Composition(formula).reduced_formula
        except (ValueError, TypeError, KeyError, OverflowError):
            rejected += 1
            continue
        vectors.append(feat.iloc[0].to_numpy(dtype=float))
        records.append({"화학식": formula, "표준 화학식": normalized, "소재 계열": str(r["Family"])})

    if not vectors:
        return pd.DataFrame(), rejected

    frame = pd.DataFrame(vectors, columns=feature_columns)
    log_vals = model.predict(frame)
    catalog = pd.DataFrame(records)
    catalog["예측 log10"] = np.asarray(log_vals, dtype=float)
    catalog["AI 예측값"] = 10.0 ** catalog["예측 log10"]
    catalog = catalog[
        np.isfinite(catalog["AI 예측값"]) & (catalog["AI 예측값"] > 0)
    ].copy()

    # 복수 문헌에 같은 조성이 반복될 수 있다. 조성별 문헌 기록 수만 계산한다.
    catalog["문헌 기록 수"] = catalog.groupby("표준 화학식")["표준 화학식"].transform("size")
    catalog = catalog.sort_values("AI 예측값", ascending=False)
    catalog = catalog.drop_duplicates(subset=["표준 화학식"], keep="first")
    return catalog.reset_index(drop=True), rejected


if selected_page == "recommend":
    st.subheader("🏆 AI 기반 전고체전해질 후보 추천")
    st.write(
        "OBELiX **공식 학습 데이터에 이미 등록된 소재** 중에서, "
        "현재 Extra Trees 모델이 예측한 이온전도도가 높은 조성을 찾아줍니다."
    )
    st.warning(
        "이 결과는 새로운 소재의 발견이나 독립 검증이 아닙니다. "
        "학습 데이터에 포함된 조성의 예측값을 이용해 후보를 정렬한 **탐색용 추천**입니다. "
        "실제 배터리 적용 가능성은 안정성·계면 특성·합성 가능성 등의 추가 검증이 필요합니다."
    )

    if st.button("🧠 추천 후보 분석 시작", use_container_width=True):
        try:
            with st.spinner("문헌 데이터를 읽고 후보들의 이온전도도를 AI로 예측하는 중..."):
                database = load_database()  # 공식 train.csv만 사용
                ranked, skipped = predict_catalog(database)
            st.session_state["ranked_catalog"] = ranked
            st.session_state["recommend_skipped"] = skipped
            st.success(f"추천 분석 완료: 서로 다른 조성 {len(ranked)}종")
        except Exception as exc:
            st.error("후보 추천을 실행하지 못했습니다. 데이터베이스나 모델을 확인해주세요.")
            st.code(str(exc))

    if "ranked_catalog" in st.session_state:
        candidates = st.session_state["ranked_catalog"]
        if candidates.empty:
            st.info("예측할 수 있는 소재가 없습니다.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                groups = sorted(g for g in candidates["소재 계열"].unique() if g.strip())
                selected_group = st.selectbox("소재 계열", ["전체"] + groups, key="recommend_family")
            with c2:
                minimum_log = st.slider(
                    "최소 예측값: log₁₀(S/cm)",
                    min_value=-12.0, max_value=-2.0,
                    value=-6.0, step=0.5,
                    help="-3은 10⁻³ S/cm, -6은 10⁻⁶ S/cm입니다.",
                )
            with c3:
                maximum_results = st.selectbox("표시할 후보 수", [5, 10, 20, 30], index=1)

            visible = candidates.copy()
            if selected_group != "전체":
                visible = visible[visible["소재 계열"] == selected_group]
            visible = visible[visible["예측 log10"] >= minimum_log]
            visible = visible.sort_values("AI 예측값", ascending=False).head(maximum_results).copy()
            st.caption(
                f"화학식 중복 제거 후 표시: {len(visible)}종. "
                f"지원하지 않거나 해석할 수 없는 자료: {st.session_state['recommend_skipped']}건."
            )

            if not visible.empty:
                visible.insert(0, "순위", range(1, len(visible) + 1))
                summary = pd.DataFrame({
                    "순위": visible["순위"].to_list(),
                    "화학식": visible["화학식"].to_list(),
                    "소재 계열": visible["소재 계열"].to_list(),
                    "AI 예측값 (S/cm)": visible["AI 예측값"].map(lambda x: f"{x:.4e}").to_list(),
                    "log₁₀(S/cm)": visible["예측 log10"].map(lambda x: f"{x:.3f}").to_list(),
                    "동일 조성 문헌 기록": visible["문헌 기록 수"].astype(int).to_list(),
                })
                v29_show_table(summary,
                    ["순위", "화학식", "AI 예측값 (S/cm)", "소재 계열"],
                    key="v29_recommend_table")

                fig_rec = px.scatter(
                    visible, x="예측 log10", y="화학식", color="소재 계열",
                    title="추천 후보의 AI 예측 이온전도도",
                    labels={"예측 log10": "예측 이온전도도 log₁₀(S/cm)", "화학식": "전해질"},
                    custom_data=["AI 예측값"],
                )
                fig_rec.update_traces(
                    marker={"size": 13},
                    hovertemplate="%{y}<br>AI 예측값: %{customdata[0]:.4e} S/cm<extra></extra>",
                )
                fig_rec.update_yaxes(autorange="reversed", automargin=True, tickfont=dict(size=12))
                fig_rec.update_layout(height=max(350, 54 * len(visible) + 130),
                                      margin=dict(l=20, r=30, t=55, b=50),
                                      font=dict(size=12))
                st.plotly_chart(fig_rec, use_container_width=True)
                st.download_button(
                    "📥 추천 결과 CSV 다운로드",
                    data=summary.to_csv(index=False).encode("utf-8-sig"),
                    file_name="SSE_AI_recommendations.csv", mime="text/csv",
                )
                st.caption(
                    "높은 이온전도도 예측값만으로 최종 전해질을 선정하지 마세요. "
                    "학습된 조성들에 대한 AI 추정치이며, 문헌 측정값이나 신규 실험의 검증 결과가 아닙니다."
                )
            else:
                st.info("현재 조건에 맞는 소재가 없습니다. 소재 계열 또는 최소 예측값을 바꿔보세요.")


# ─────────────────────────────────────────────────────────────────────────────
# 네 번째 메뉴: 전역 특징 중요도와 "실제 모델 학습 구간" 조성 유사도
# ─────────────────────────────────────────────────────────────────────────────
def get_training_reference():
    """Colab에서 직접 내보낸 실제 학습용 CSV를 이용한다. 분할을 추측하지 않음."""
    if not TRAIN_REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            "SSE_train_reference.csv 파일을 app.py와 같은 폴더에 넣어주세요. "
            "Google Colab에서 df_ml.loc[X_train.index]로 저장한 파일이어야 합니다."
        )
    raw = pd.read_csv(TRAIN_REFERENCE_PATH, dtype={FORMULA_COL: str})
    required = {FORMULA_COL, "conductivity", "log_conductivity"}
    if not required.issubset(raw.columns):
        raise ValueError(f"학습 CSV에 필요한 열이 없습니다: {sorted(required - set(raw.columns))}")

    family_map = {}
    if DATABASE_PATH.is_file():
        try:
            meta = pd.read_csv(DATABASE_PATH, usecols=[FORMULA_COL, "Family"], dtype=str).fillna("")
            family_map = meta.drop_duplicates(FORMULA_COL).set_index(FORMULA_COL)["Family"].to_dict()
        except Exception:
            pass

    metadata, vectors = [], []
    skipped = 0
    for _, record in raw.iterrows():
        formula = str(record[FORMULA_COL]).strip()
        try:
            sample = make_input(formula)
            normalized = Composition(formula).reduced_formula
        except (ValueError, TypeError, KeyError, OverflowError):
            skipped += 1
            continue
        metadata.append({
            "화학식": formula,
            "표준 화학식": normalized,
            "소재 계열": family_map.get(formula, "미분류"),
            "학습 측정값 (S/cm)": record["conductivity"],
        })
        vectors.append(sample.iloc[0].to_numpy(dtype=float))

    if not vectors:
        raise ValueError("학습 참조 CSV에 분석 가능한 화학식이 없습니다.")
    table = pd.DataFrame(metadata)
    frame = pd.DataFrame(vectors, columns=feature_columns)
    return table, frame, len(raw), skipped


if selected_page == "analysis":
    st.subheader("📊 AI 모델 분석")
    st.write("현재 사용 중인 Extra Trees 모델의 **전체 특징 중요도**와 입력 소재의 **학습 조성 유사도**를 확인합니다.")

    st.markdown("#### 1. 모델이 중요하게 활용한 입력 특징")
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    weights = getattr(estimator, "feature_importances_", None)

    if weights is None or len(weights) != len(feature_columns):
        st.info("현재 모델은 내장 특징 중요도를 제공하지 않습니다.")
    else:
        importance = pd.DataFrame({
            "원소": feature_columns,
            "중요도": np.asarray(weights, dtype=float),
        }).sort_values("중요도", ascending=False)
        n_top = st.slider("표시할 원소 수", 5, min(30, len(feature_columns)), 15, step=5)
        leading = importance.head(n_top).sort_values("중요도", ascending=True)
        fig_imp = px.bar(
            leading, x="중요도", y="원소", orientation="h",
            title="Extra Trees의 전체 모델 특징 중요도",
            labels={"중요도": "모델 내부 중요도 (상대적 수치)", "원소": "원소 조성비"},
        )
        fig_imp.update_layout(height=max(350, 28 * n_top + 120), margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "이 그래프는 트리 분할에서 계산된 '모델 전체의 불순도 기반 특징 중요도'입니다. "
            "개별 소재의 예측 이유나 물리적 인과관계, 실험 효과를 증명하는 SHAP 결과는 아닙니다. "
            "서로 상관된 특징과 원소의 등장 빈도에 영향을 받을 수 있습니다."
        )

    st.divider()
    st.markdown("#### 2. 입력 화학식과 실제 학습 조성 비교")
    st.write(
        "Colab에서 직접 저장한 354개 학습 참조 자료에서 조성비가 가까운 소재를 찾습니다. "
        "현재 모델은 결정구조가 아니라 원소 비율 52개를 입력으로 사용하므로, "
        "서로 다른 결정상도 동일한 조성으로 취급할 수 있습니다."
    )
    with st.form("similarity_form"):
        query_formula = st.text_input(
            "분석할 전해질 화학식", value="Li6PS5Cl", key="analysis_formula",
            help="예: Li6PS5Cl, Li3PS4, Li7La3Zr2O12",
        )
        neighbor_count = st.selectbox("표시할 유사 소재 수", [5, 8, 10], index=0)
        run_similarity = st.form_submit_button("🔍 학습 소재와 비교하기", use_container_width=True)

    if run_similarity:
        try:
            with st.spinner("유사한 학습 화학식을 찾는 중..."):
                sample = make_input(query_formula)
                train_meta, train_mat, exported_n, skipped_n = get_training_reference()

            # 원소 원자분율은 합계가 1이므로 총변동거리(TV) = 0.5 * L1.
            # 유사도 = 100 * (1 - TV) (0~100%), 예측 정확도와 무관함.
            q_vector = sample.iloc[0].to_numpy(dtype=float)
            dist = 0.5 * np.abs(train_mat.to_numpy(dtype=float) - q_vector).sum(axis=1)
            similarity = 100.0 * (1.0 - np.clip(dist, 0.0, 1.0))

            nearby = train_meta.copy()
            nearby["조성 유사도 (%)"] = similarity
            nearby["조성 거리"] = dist
            nearby["같은 화학식의 학습 기록 수"] = nearby.groupby("표준 화학식")["표준 화학식"].transform("size")
            nearby = nearby.sort_values("조성 거리", ascending=True)
            nearby = nearby.drop_duplicates("표준 화학식").head(neighbor_count)

            pred = predict_one(query_formula)
            a, b, c = st.columns(3)
            a.metric("AI 예측 이온전도도 (S/cm)", f"{pred['conductivity_S_per_cm']:.3e}")
            b.metric("가장 가까운 조성 유사도", f"{nearby['조성 유사도 (%)'].iloc[0]:.1f}%")
            c.metric("분석 기준 학습 기록", f"{len(train_meta)}건")

            st.caption(
                f"Colab에서 내보낸 학습 CSV {exported_n}행 중 분석에 사용된 {len(train_meta)}행. "
                "이 목록은 공식 원본에서 분할을 추정한 것이 아니라, 직접 내보낸 학습 참조 파일에 기반합니다. "
                "다만 해당 파일이 현재 모델을 학습한 실행과 동일한지 자동으로 증명할 수는 없습니다."
            )
            if exported_n != 354 or skipped_n:
                st.warning(
                    f"기존 실습에서 예상한 학습 데이터 354건과 차이가 있습니다. "
                    f"CSV {exported_n}행, 화학식 파싱 제외 {skipped_n}행. "
                    "Colab에서 같은 모델 학습 실행으로 저장한 파일인지 확인하세요."
                )

            shown = nearby[["화학식", "소재 계열", "조성 유사도 (%)", "같은 화학식의 학습 기록 수"]].copy()
            shown["조성 유사도 (%)"] = shown["조성 유사도 (%)"].map(lambda x: f"{x:.1f}")
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.info(
                "조성 유사도 100%는 원소 비율이 같다는 뜻이지, 예측 정확도 100% 또는 같은 결정구조라는 뜻이 아닙니다. "
                "유사도가 높아도 실제 측정값과 AI 예측값 사이에 오차가 클 수 있습니다."
            )
            st.caption(
                "유사도 계산: 100 × [1 − ½ × Σ|입력 원소분율 − 학습 소재 원소분율|]. "
                "조성만 비교하는 탐색 지표이며 통계적 예측 신뢰구간이나 검증된 정확도 확률이 아닙니다."
            )

        except Exception as exc:
            st.error("조성 유사도 분석에 실패했습니다. 데이터베이스 연결 또는 화학식을 확인해주세요.")
            st.code(str(exc))
            st.caption("처음 분석할 때는 OBELiX 공식 학습 데이터를 다운로드할 수 있도록 인터넷 연결이 필요합니다.")

if selected_page == "validation":
    st.subheader("📈 AI 모델 성능 검증")
    st.caption("내부 검증·공식 테스트·교차검증은 평가 자료와 목적이 다르므로 숫자만으로 모델 순위를 판단하지 않습니다.")
    vcol1, vcol2, vcol3 = st.columns(3)
    vcol1.metric("과거 내부 검증 MAE", "0.596", help="기존 v1 모델 · 개발 과정의 95건")
    vcol2.metric("과거 공식 테스트 MAE", "1.522", help="기존 v1 모델 · 별도 평가 가능 109건")
    vcol3.metric("교차검증 A 모델 MAE", "1.289", help="개발용 449건 · 조성+DOI 연결 분리")
    st.caption("세 수치는 서로 다른 검증 절차의 기록이며 동일한 테스트 데이터에서 얻은 직접 비교 결과가 아닙니다.")
    validation_view = st.radio(
        "확인할 검증 결과", ["cv", "internal"],
        format_func=lambda v: "A·B·C 교차검증" if v == "cv" else "기존 모델 내부 검증 · 공식 테스트 설명",
        horizontal=True, key="v293_validation_view",
    )
    st.divider()

if selected_page == "validation" and validation_view == "internal":
    st.subheader("🧪 검증 데이터의 실제값과 AI 예측값 비교")
    st.write("개발 기록 449건을 354건 학습·95건 검증으로 나누어, 기존 모델의 실험값과 예측값을 비교합니다.")
    st.caption("내부 검증 MAE 0.596 · R² 0.776 / 별도 공식 테스트(평가 가능 109건) MAE 1.522 · R² −0.084")
    with st.expander("내부 검증과 공식 테스트의 차이 · 지표 설명", expanded=False):
        st.write("**내부 검증:** 개발 과정에서 모델을 고를 때 참고한 95건의 평가입니다. 최종 독립 테스트가 아닙니다.")
        st.write("**공식 테스트:** OBELiX의 별도 테스트 121건 중 평가 가능한 109건에 대해 기존 v1을 평가했던 기록입니다. 이 화면에서 재실행하지 않으며 v2의 공식 테스트는 미평가입니다.")
        st.write("**MAE(log₁₀):** 실험값과 예측값 사이의 평균 절대 로그 오차이며 낮을수록 좋습니다. **R²:** 일반적으로 높을수록 좋고 음수일 수도 있습니다.")
        st.warning("서로 다른 평가 데이터이므로 MAE 차이의 원인을 수치만으로 단정할 수 없으며, 내부 검증으로 신규 소재의 정확도를 보장할 수 없습니다.")

    if not VALID_REFERENCE_PATH.is_file():
        st.info(
            "아직 `SSE_valid_reference.csv`가 없습니다. "
            "Colab에서 `df_ml.loc[X_valid.index]`로 내보내 앱 폴더에 복사해주세요. "
            "기존 예측·검색·추천·유사도 기능은 그대로 사용할 수 있습니다."
        )
        st.code(
            'from google.colab import files\n'
            'valid_reference = df_ml.loc[X_valid.index, ["Reduced Composition", "conductivity", "log_conductivity"]].copy()\n'
            'valid_reference.to_csv("SSE_valid_reference.csv", index=False, encoding="utf-8-sig")\n'
            'print("검증 자료:", len(valid_reference))\n'
            'files.download("SSE_valid_reference.csv")',
            language="python",
        )
    elif st.button("🔍 검증 데이터 95개 분석하기", use_container_width=True):
        try:
            raw_valid = pd.read_csv(VALID_REFERENCE_PATH, dtype={FORMULA_COL: str})
            required = {FORMULA_COL, "conductivity", "log_conductivity"}
            if not required.issubset(raw_valid.columns):
                raise ValueError(f"필수 열이 없습니다: {sorted(required - set(raw_valid.columns))}")
            records, vectors = [], []
            skipped = 0
            for _, record in raw_valid.iterrows():
                formula = str(record[FORMULA_COL]).strip()
                actual = pd.to_numeric(record["log_conductivity"], errors="coerce")
                if not np.isfinite(actual):
                    skipped += 1
                    continue
                try:
                    vector = make_input(formula)
                except (ValueError, TypeError, KeyError, OverflowError):
                    skipped += 1
                    continue
                vectors.append(vector.iloc[0].to_numpy(dtype=float))
                records.append({"화학식": formula, "실제 log10": float(actual)})
            if not vectors:
                raise ValueError("예측 가능한 검증 데이터가 없습니다.")
            pred_frame = pd.DataFrame(vectors, columns=feature_columns)
            predictions = model.predict(pred_frame)
            eval_df = pd.DataFrame(records)
            eval_df["예측 log10"] = predictions
            eval_df["절대오차 (log10)"] = (eval_df["실제 log10"] - eval_df["예측 log10"]).abs()
            eval_df["문헌 측정값 (S/cm)"] = 10.0 ** eval_df["실제 log10"]
            eval_df["AI 예측값 (S/cm)"] = 10.0 ** eval_df["예측 log10"]
            st.session_state["validation_evaluation"] = (eval_df, len(raw_valid), skipped)
        except Exception as exc:
            st.error("검증 데이터를 분석하지 못했습니다. 파일과 Colab 분할을 확인해주세요.")
            st.code(str(exc))

    if "validation_evaluation" in st.session_state:
        eval_df, n_rows, skipped = st.session_state["validation_evaluation"]
        mae = mean_absolute_error(eval_df["실제 log10"], eval_df["예측 log10"])
        r2 = r2_score(eval_df["실제 log10"], eval_df["예측 log10"]) if len(eval_df) > 1 else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("평균 절대오차 MAE (log10)", f"{mae:.3f}")
        c2.metric("결정계수 R²", f"{r2:.3f}")
        c3.metric("검증에 사용한 기록", f"{len(eval_df)} / {n_rows}건")
        if n_rows != 95 or skipped:
            st.warning(f"기존 실습의 95개 검증행과 차이가 있습니다. 원본 {n_rows}건, 제외 {skipped}건.")
        else:
            st.caption("같은 모델·데이터로 내보냈다면 기존 결과 MAE 약 0.596, R² 약 0.776에 가까워야 합니다.")

        fig_v = px.scatter(
            eval_df, x="실제 log10", y="예측 log10", hover_data=["화학식"],
            title="검증 데이터: 실제값과 예측값 비교",
            labels={"실제 log10": "실제 이온전도도 log₁₀(S/cm)", "예측 log10": "AI 예측 log₁₀(S/cm)"},
        )
        lo = min(eval_df["실제 log10"].min(), eval_df["예측 log10"].min())
        hi = max(eval_df["실제 log10"].max(), eval_df["예측 log10"].max())
        fig_v.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                        line=dict(color="gray", dash="dash", width=2))
        st.plotly_chart(fig_v, use_container_width=True)
        with st.expander("오차가 큰 소재 TOP 10", expanded=False):
            st.markdown("#### 예측 오차가 큰 소재 TOP 10")
            top_errors = eval_df.sort_values("절대오차 (log10)", ascending=False).head(10)
            top_view = top_errors.copy()
            # 원본 수치는 그대로 보존하고 화면에 보여주는 값만 지수 표기로 변환한다.
            for field in ["문헌 측정값 (S/cm)", "AI 예측값 (S/cm)"]:
                top_view[field] = top_view[field].map(lambda x: f"{x:.3e}")
            for field in ["실제 log10", "예측 log10", "절대오차 (log10)"]:
                top_view[field] = top_view[field].map(lambda x: f"{x:.3f}")
            st.dataframe(top_view, use_container_width=True, hide_index=True)
            st.caption("표의 이온전도도는 과학적 표기법으로 표시했습니다. CSV에는 계산에 사용한 원본 숫자가 저장됩니다.")
        with st.expander("검증 데이터 결과 저장", expanded=False):
            st.download_button(
                "📥 검증 결과 CSV 다운로드",
                data=eval_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_AI_validation_results.csv", mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# 여섯 번째 메뉴: 기존 52개 모델 vs 물리특성 59개 모델
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == "compare":
    st.subheader("⚖️ 두 AI 모델 비교")
    st.write("동일한 화학식을 **52개 원소 조성비 모델(v1)**과 **59개 특징 모델(v2)**에 입력하여 결과를 비교합니다.")
    st.caption("v2는 원소 조성비에 원자번호·질량·전기음성도·원자 반경 등 7개 특징을 더한 Extra Trees 모델입니다.")

    if not MODEL_V2_PATH.is_file():
        st.error("`SSE_AI_Model_v2.joblib` 파일이 없습니다. app.py와 같은 폴더에 넣어주세요.")
    else:
        try:
            model_v2, feature_columns_v2 = load_model_v2()
        except Exception as exc:
            st.error("59개 특징 모델을 불러오지 못했습니다.")
            st.code(str(exc))
            st.caption("모델을 저장한 Colab과 PC의 scikit-learn 버전 차이를 확인해주세요.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("v1 입력 특징", f"{len(feature_columns)}개")
            c2.metric("v2 입력 특징", f"{len(feature_columns_v2)}개")

            with st.form("compare_models_form"):
                compare_formula = st.text_input(
                    "비교할 고체전해질 화학식", value="Li6PS5Cl",
                    help="Li6PS5Cl, Li3PS4, Li7La3Zr2O12 등을 입력해보세요.",
                    key="dual_formula",
                )
                compare_click = st.form_submit_button("🧠 두 AI 모델로 동시에 예측", use_container_width=True)

            if compare_click:
                try:
                    row1 = make_input(compare_formula)
                    row2 = make_input_v2(compare_formula, feature_columns_v2)
                    log1 = float(model.predict(row1)[0])
                    log2 = float(model_v2.predict(row2)[0])
                    if not np.isfinite(log1) or not np.isfinite(log2):
                        raise ValueError("유효하지 않은 AI 예측값이 나왔습니다.")
                    st.session_state["dual_result"] = {
                        "formula": normalize_user_formula(compare_formula),
                        "v1_log": log1, "v2_log": log2,
                    }
                except Exception as exc:
                    st.session_state.pop("dual_result", None)
                    st.error(f"두 모델 비교를 진행하지 못했습니다: {exc}")
                    st.info(v29_formula_error_hint(compare_formula, exc))

            if "dual_result" in st.session_state:
                result = st.session_state["dual_result"]
                v1_s = 10.0 ** result["v1_log"]
                v2_s = 10.0 ** result["v2_log"]
                st.success(f"{result['formula']}의 두 모델 예측 완료")
                a,b = st.columns(2)
                a.metric("v1 예측값 (S/cm)", f"{v1_s:.4e}")
                b.metric("v2 예측값 (S/cm)", f"{v2_s:.4e}")
                st.caption(f"v1 log₁₀: {result['v1_log']:.4f} · v2 log₁₀: {result['v2_log']:.4f}")
                compare_df = pd.DataFrame({
                    "모델": ["v1 · 52개", "v2 · 59개"],
                    "log10(S/cm)": [result["v1_log"], result["v2_log"]],
                    "예측 이온전도도 (S/cm)": [v1_s, v2_s],
                })
                relative_pct = (v2_s / v1_s - 1.0) * 100.0
                st.metric("v2 예측값의 v1 대비 차이", f"{relative_pct:+.1f}%")
                st.caption("두 모델의 **출력값 차이**입니다. 예측 정확도나 통계적 신뢰도로 해석하면 안 됩니다.")

                chart_mode = st.radio(
                    "비교 그래프 표시 방식",
                    ["0에서 시작하는 막대그래프", "로그축 점그래프"],
                    index=0, horizontal=True, key="v7_chart_mode",
                )
                if chart_mode == "0에서 시작하는 막대그래프":
                    fig_cmp = px.bar(
                        compare_df, x="모델", y="예측 이온전도도 (S/cm)",
                        text=compare_df["예측 이온전도도 (S/cm)"].map(lambda v: f"{v:.3e}"),
                        title=f"{result['formula']} — 두 AI 모델 예측값 (0 기준)",
                    )
                    fig_cmp.update_yaxes(
                        type="linear", range=[0, float(max(v1_s, v2_s) * 1.28)],
                        tickformat=".2e", zeroline=True,
                    )
                    fig_cmp.update_traces(textposition="outside", cliponaxis=False)
                else:
                    fig_cmp = px.scatter(
                        compare_df, x="log10(S/cm)", y="모델",
                        text=compare_df["예측 이온전도도 (S/cm)"].map(lambda v: f"{v:.3e}"),
                        title=f"{result['formula']} — 로그축 예측값 비교",
                        labels={"log10(S/cm)": "예측 이온전도도 log₁₀(S/cm)"},
                        custom_data=["예측 이온전도도 (S/cm)"],
                    )
                    fig_cmp.update_traces(
                        mode="markers+text", textposition="middle right",
                        marker={"size": 17},
                        hovertemplate="%{y}<br>예측값: %{customdata[0]:.4e} S/cm<extra></extra>",
                    )
                    fig_cmp.update_xaxes(range=[float(min(result["v1_log"], result["v2_log"]) - 0.25),
                                                float(max(result["v1_log"], result["v2_log"]) + 0.55)])
                fig_cmp.update_layout(height=385, margin=dict(l=10, r=20, t=55, b=40))
                st.plotly_chart(fig_cmp, use_container_width=True)
                st.info(
                    "두 모델이 비슷한 값을 예측해도 맞다는 보장은 없습니다. "
                    "**모델 간 차이 ≠ 검증된 불확실성 구간/정확도**입니다."
                )

                st.markdown("#### 🔎 학습 조성 유사도 확인 (참고 지표)")
                try:
                    reference_meta, reference_matrix, reference_total, reference_skipped = get_training_reference()
                    query_vector = make_input(result["formula"]).iloc[0].to_numpy(dtype=float)
                    distances = 0.5 * np.abs(reference_matrix.to_numpy(dtype=float) - query_vector).sum(axis=1)
                    if len(distances) == 0:
                        raise ValueError("분석 가능한 학습 조성이 없습니다.")
                    nearest_index = int(np.argmin(distances))
                    nearest_pct = 100 * (1 - float(np.clip(distances[nearest_index], 0, 1)))
                    exact_count = int(np.sum(np.isclose(distances, 0, atol=1e-10)))
                    m1, m2 = st.columns(2)
                    m1.metric("가장 가까운 학습 조성 유사도", f"{nearest_pct:.1f}%")
                    m2.metric("동일 조성 학습 기록", f"{exact_count}건")
                    st.caption(
                        f"가장 가까운 소재: {reference_meta.iloc[nearest_index]['화학식']} · "
                        f"Colab에서 내보낸 학습 참조 CSV {reference_total}건 사용"
                    )
                    if exact_count:
                        st.warning(
                            "입력한 조성과 동일한 학습 기록이 있습니다. 이 결과는 학습 분포 내 예측일 수 있으며, "
                            "새로운 화학식에 대한 독립적 예측 성능을 입증하지 않습니다."
                        )
                    else:
                        st.info(
                            "학습 참조 파일에서 동일한 조성을 찾지 못했습니다. "
                            "가장 가까운 조성과의 거리만 참고하고, 예측을 실험값처럼 사용하지 마세요."
                        )
                    st.caption(
                        "조성 유사도는 원소 원자분율 차이에 기반한 탐색 지표입니다. "
                        "정확도 %, 신뢰도 확률 또는 통계적으로 보정된 예측 구간이 아닙니다."
                    )
                except Exception as exc:
                    st.info(
                        "학습 소재 유사도를 확인하려면 `SSE_train_reference.csv`가 필요합니다. "
                        "예측값 비교는 그대로 이용할 수 있습니다."
                    )
                    st.caption(f"유사도 분석 안내: {exc}")
                st.download_button(
                    "📥 두 모델 예측 결과 CSV 저장",
                    compare_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="SSE_AI_dual_predictions.csv", mime="text/csv",
                )

            st.divider()
            st.markdown("#### 모델 평가 결과 비교")
            st.write("앞서 Colab에서 기록했던 **95개 내부 검증 결과**와 **354개 학습 자료의 계열 기반 교차검증**을 비교합니다.")
            previous = pd.DataFrame({
                "모델": ["v1 · 52개", "v2 · 59개"],
                "내부 검증 MAE (95건)": [0.596, 0.617],
                "내부 검증 R² (95건)": [0.776, 0.753],
                "계열 기반 CV MAE (354건)": [2.168, 1.717],
            })
            st.dataframe(previous, use_container_width=True, hide_index=True)
            st.caption("과거 Colab 실행 결과를 기재한 표입니다. 내부 검증은 모델 선택에 사용했으므로 독립 최종 평가가 아닙니다. 공식 테스트셋으로 튜닝하지 마세요.")

            if VALID_REFERENCE_PATH.is_file():
                if st.button("🔍 95개 검증 데이터로 두 모델 성능 다시 계산", use_container_width=True):
                    try:
                        valid_raw = pd.read_csv(VALID_REFERENCE_PATH, dtype={FORMULA_COL: str})
                        expected = {FORMULA_COL, "log_conductivity"}
                        if not expected.issubset(valid_raw.columns):
                            raise ValueError("검증 CSV에 필요한 열이 없습니다.")
                        feat1, feat2, targets = [], [], []
                        skipped = 0
                        for _, rec in valid_raw.iterrows():
                            try:
                                y_true = float(rec["log_conductivity"])
                                if not np.isfinite(y_true):
                                    skipped += 1
                                    continue
                                inp1 = make_input(str(rec[FORMULA_COL]))
                                inp2 = make_input_v2(str(rec[FORMULA_COL]), feature_columns_v2)
                                feat1.append(inp1.iloc[0].to_numpy(dtype=float))
                                feat2.append(inp2.iloc[0].to_numpy(dtype=float))
                                targets.append(y_true)
                            except (ValueError, TypeError, KeyError, OverflowError):
                                skipped += 1
                        if not targets:
                            raise ValueError("평가 가능한 검증 소재가 없습니다.")
                        x1 = pd.DataFrame(feat1, columns=feature_columns)
                        x2 = pd.DataFrame(feat2, columns=feature_columns_v2)
                        p1 = model.predict(x1)
                        p2 = model_v2.predict(x2)
                        st.session_state["dual_validation"] = {
                            "n": len(targets), "skipped": skipped,
                            "scores": pd.DataFrame({
                                "모델": ["v1 · 52개", "v2 · 59개"],
                                "MAE": [mean_absolute_error(targets, p1), mean_absolute_error(targets, p2)],
                                "R²": [r2_score(targets, p1), r2_score(targets, p2)],
                            }),
                        }
                    except Exception as exc:
                        st.error(f"95개 검증 데이터의 비교에 실패했습니다: {exc}")

                if "dual_validation" in st.session_state:
                    validation = st.session_state["dual_validation"]
                    st.caption(f"검증에 사용한 기록: {validation['n']}건 · 제외: {validation['skipped']}건")
                    st.dataframe(validation["scores"].round(3), use_container_width=True, hide_index=True)
            else:
                st.info("95개 검증 결과를 앱에서 다시 계산하려면 `SSE_valid_reference.csv` 파일이 필요합니다.")

            st.divider()
            st.markdown("#### 📋 공식 테스트 평가 기록 (v1만)")
            st.write(
                "아래는 Colab에서 **기존 v1 모델(52개 특징)**로 이미 수행했던 "
                "OBELiX 공식 테스트 결과를 *기록한 표*입니다. 이 메뉴에서 테스트 데이터를 "
                "다시 불러와 평가하지 않으며, v2에 대한 공식 테스트 성능도 표시하지 않습니다."
            )
            official = pd.DataFrame({
                "평가 범위": ["공식 테스트 평가 가능 109건", "학습 조성과 중복 2건 제외 107건"],
                "평가 데이터 수": [109, 107],
                "v1 MAE (log₁₀)": [1.522, 1.545],
                "v1 R²": [-0.084, -0.095],
                "v2 공식 테스트": ["미평가", "미평가"],
            })
            st.dataframe(official, use_container_width=True, hide_index=True)
            st.warning(
                "이 결과는 95개 내부 검증(MAE 0.596)보다 나쁩니다. "
                "공식 테스트 결과를 이미 확인하고 오차 특성까지 분석했으므로, "
                "이후 모델을 선택·수정하는 반복 과정에 재사용하지 마세요. "
                "v2 모델의 성능이 개선됐다는 독립 테스트 근거는 아직 없습니다."
            )
            st.caption(
                "전체 공식 테스트 121건 중 양의 수치형 113건, "
                "그중 지원되지 않는 화학식 4건을 제외한 109건 평가. "
                "조성 중복 2건을 추가로 제외한 평가는 107건입니다. "
                "모두 이전 Colab 분석에서 기록한 수치입니다."
            )
            report = pd.DataFrame({
                "모델": ["v1 · 52개", "v2 · 59개"],
                "내부 검증 MAE (95건)": [0.596, 0.617],
                "내부 검증 R² (95건)": [0.776, 0.753],
                "계열 기반 CV MAE (354건)": [2.168, 1.717],
                "공식 테스트 MAE (109건, v1만)": [1.522, np.nan],
                "공식 테스트 R² (109건, v1만)": [-0.084, np.nan],
            })
            st.download_button(
                "📥 모델 평가 요약 CSV 다운로드",
                report.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_AI_model_evaluation_summary.csv", mime="text/csv",
                help="기존 Colab 평가 기록입니다. 새로운 테스트를 자동 실행하는 기능은 아닙니다.",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 일곱 번째 메뉴 (v8): Li6PS5Cl(1-x)Br(x) 치환 조성 시뮬레이션
# ─────────────────────────────────────────────────────────────────────────────
def make_halide_formula(br_fraction: float) -> str:
    """Li6PS5Cl(1-x)Br(x). 0 계수의 원소는 화학식에서 제거한다."""
    x = round(float(br_fraction), 2)
    if not 0.0 <= x <= 1.0:
        raise ValueError("Br 비율은 0~1 사이여야 합니다.")

    def coefficient(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    base = "Li6PS5"
    if x < 1.0:
        base += "Cl" + ("" if x == 0.0 else coefficient(1.0 - x))
    if x > 0.0:
        base += "Br" + ("" if x == 1.0 else coefficient(x))
    return base


def run_halide_sweep(model_v2, v2_columns):
    """미리 정한 21개 조성을 두 모델에 전달; 원래 입력 특징 순서 유지."""
    fractions = np.round(np.linspace(0.0, 1.0, 21), 2)
    formulas = [make_halide_formula(x) for x in fractions]
    frame_v1 = pd.concat([make_input(f) for f in formulas], ignore_index=True)
    frame_v2 = pd.concat([make_input_v2(f, v2_columns) for f in formulas], ignore_index=True)

    log_v1 = np.asarray(model.predict(frame_v1), dtype=float)
    log_v2 = np.asarray(model_v2.predict(frame_v2), dtype=float)
    conductivity_v1 = 10.0 ** log_v1
    conductivity_v2 = 10.0 ** log_v2

    if (not np.isfinite(conductivity_v1).all() or
            not np.isfinite(conductivity_v2).all() or
            (conductivity_v1 <= 0).any() or (conductivity_v2 <= 0).any()):
        raise ValueError("일부 조성의 AI 예측값이 유효하지 않습니다.")

    return pd.DataFrame({
        "Br 치환 비율 (%)": (fractions * 100).astype(int),
        "Cl 비율": np.round(1.0 - fractions, 2),
        "Br 비율": fractions,
        "화학식": formulas,
        "v1 log10(S/cm)": log_v1,
        "v2 log10(S/cm)": log_v2,
        "v1 예측 이온전도도 (S/cm)": conductivity_v1,
        "v2 예측 이온전도도 (S/cm)": conductivity_v2,
    })


if selected_page == "_legacy_halide":
    st.subheader("🧪 Cl/Br 치환 비율에 따른 이온전도도 시뮬레이션")
    st.write(
        "**Li₆PS₅Cl₁₋ₓBrₓ**의 Cl을 Br로 치환하는 비율을 바꿔가며 "
        "**v1(52개 특징)**과 **v2(59개 특징)**의 예측 변화를 비교합니다."
    )
    st.warning(
        "가상 조성을 화학식 수준에서 탐색하는 기능입니다. "
        "실제로 합성 가능하거나 단일상을 형성한다는 보장이 없으며, "
        "예측 최고점은 **실험으로 검증된 최적 조성**이 아닙니다. "
        "결정구조·합성법·온도·계면 안정성은 두 모델 모두 직접 반영하지 않습니다."
    )

    if not MODEL_V2_PATH.is_file():
        st.error("두 번째 모델 파일 `SSE_AI_Model_v2.joblib`이 없습니다. app.py와 같은 폴더에 넣어주세요.")
    else:
        try:
            sweep_model_v2, sweep_columns_v2 = load_model_v2()
        except Exception as exc:
            st.error("v2 모델을 열지 못했습니다. 파일과 scikit-learn 버전을 확인해주세요.")
            st.code(str(exc))
        else:
            x_percent = st.slider(
                "Br 치환 비율 (%)", min_value=0, max_value=100,
                value=50, step=5, key="v8_br_fraction",
            )
            x_fraction = x_percent / 100.0
            formula_now = make_halide_formula(x_fraction)
            s1, s2, s3 = st.columns(3)
            s1.metric("Cl 함량 (1−x)", f"{1.0-x_fraction:.2f}")
            s2.metric("Br 함량 (x)", f"{x_fraction:.2f}")
            s3.metric("입력한 Br 치환 비율", f"{x_percent}%")
            st.code(formula_now, language="text")
            st.caption("슬라이더를 움직이면 조성이 바뀝니다. 아래 버튼을 눌러 0~100%를 5% 간격으로 한 번에 분석하세요.")

            if st.button("🔬 두 AI 모델로 21개 조성 분석하기", use_container_width=True, key="v8_run"):
                try:
                    with st.spinner("21개 조성의 이온전도도를 예측하고 있습니다..."):
                        st.session_state["v8_sweep_results"] = run_halide_sweep(
                            sweep_model_v2, sweep_columns_v2
                        )
                except Exception as exc:
                    st.session_state.pop("v8_sweep_results", None)
                    st.error("조성 시뮬레이션을 수행하지 못했습니다.")
                    st.code(str(exc))

            if "v8_sweep_results" in st.session_state:
                result_sweep = st.session_state["v8_sweep_results"]
                selected = result_sweep.loc[
                    result_sweep["Br 치환 비율 (%)"] == x_percent
                ]
                if selected.empty:
                    st.error("선택한 조성에 대한 결과를 찾지 못했습니다.")
                else:
                    point = selected.iloc[0]
                    st.markdown("#### 선택한 치환 비율의 AI 예측값")
                    c1, c2 = st.columns(2)
                    c1.metric("v1 · 52개 특징 (S/cm)", f"{point['v1 예측 이온전도도 (S/cm)']:.4e}")
                    c2.metric("v2 · 59개 특징 (S/cm)", f"{point['v2 예측 이온전도도 (S/cm)']:.4e}")
                    st.caption(
                        f"선택된 화학식: {point['화학식']} · "
                        f"v1 log₁₀={point['v1 log10(S/cm)']:.3f}, "
                        f"v2 log₁₀={point['v2 log10(S/cm)']:.3f}"
                    )

                st.markdown("#### Cl/Br 치환 비율에 따른 AI 예측 변화")
                curves = result_sweep.melt(
                    id_vars=["Br 치환 비율 (%)", "화학식"],
                    value_vars=["v1 log10(S/cm)", "v2 log10(S/cm)"],
                    var_name="AI 모델", value_name="예측 log₁₀(S/cm)"
                )
                fig_sweep = px.line(
                    curves, x="Br 치환 비율 (%)", y="예측 log₁₀(S/cm)",
                    color="AI 모델", markers=True, custom_data=["화학식"],
                    title="Cl → Br 치환에 따른 예측 이온전도도 변화",
                )
                fig_sweep.add_vline(
                    x=x_percent, line_dash="dot", line_color="gray",
                    annotation_text="현재 선택", annotation_position="top",
                )
                fig_sweep.update_traces(
                    hovertemplate="Br 치환: %{x:.0f}%<br>예측 log₁₀: %{y:.3f}<br>화학식: %{customdata[0]}<extra>%{fullData.name}</extra>"
                )
                fig_sweep.update_layout(height=440, hovermode="x unified", legend_title_text="모델")
                st.plotly_chart(fig_sweep, use_container_width=True)
                st.caption(
                    "세로축은 log₁₀(S/cm)입니다. 1 증가하면 이온전도도는 10배입니다. "
                    "이 곡선은 21개 조성의 모델 추정값을 연결한 것이며 실제 측정 곡선은 아닙니다."
                )

                st.markdown("#### 모델별 예측 최고점 · 탐색용")
                opt1 = result_sweep.loc[result_sweep["v1 예측 이온전도도 (S/cm)"].idxmax()]
                opt2 = result_sweep.loc[result_sweep["v2 예측 이온전도도 (S/cm)"].idxmax()]
                r1, r2 = st.columns(2)
                with r1:
                    st.metric("v1 최고 예측 조성 (Br)", f"{int(opt1['Br 치환 비율 (%)'])}%")
                    st.write(f"{opt1['화학식']} · {opt1['v1 예측 이온전도도 (S/cm)']:.3e} S/cm")
                with r2:
                    st.metric("v2 최고 예측 조성 (Br)", f"{int(opt2['Br 치환 비율 (%)'])}%")
                    st.write(f"{opt2['화학식']} · {opt2['v2 예측 이온전도도 (S/cm)']:.3e} S/cm")
                st.info(
                    "위 최고점은 **5% 간격 21개 가상 조성 중 해당 모델이 가장 높게 예측한 값**입니다. "
                    "두 모델이 같은 결과를 내더라도 정확도가 입증되는 것은 아닙니다."
                )

                st.markdown("#### 전체 21개 조성 비교표")
                view_sweep = result_sweep.copy()
                for col in ["v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"]:
                    view_sweep[col] = view_sweep[col].map(lambda n: f"{n:.4e}")
                for col in ["v1 log10(S/cm)", "v2 log10(S/cm)"]:
                    view_sweep[col] = view_sweep[col].map(lambda n: f"{n:.3f}")
                v29_show_table(view_sweep,
                    ["Br 치환 비율 (%)", "화학식", "v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"],
                    key="v29_sweep_table")
                st.download_button(
                    "📥 조성 시뮬레이션 결과 CSV 다운로드",
                    data=result_sweep.to_csv(index=False).encode("utf-8-sig"),
                    file_name="SSE_AI_v8_ClBr_substitution.csv", mime="text/csv",
                    key="v8_download",
                )
            else:
                st.info("위 버튼을 누르면 두 AI 모델의 조성별 예측 그래프와 비교표가 나타납니다.")



# ─────────────────────────────────────────────────────────────────────────────
# v29.2: 범용 원소 치환(동일 원자 자리의 형식적 1:1 치환) 탐색 기능.
# 기존 Li6PS5Cl1-xBrx 시뮬레이션과 저장 모델/교차검증 파일은 변경하지 않습니다.
# ─────────────────────────────────────────────────────────────────────────────

def v292_substitution_compositions(base_formula: str, source: str, target: str) -> pd.DataFrame:
    """0~100%를 5%씩 치환한 21개 화학식 생성. 실제 결정상 또는 전하중성 검증은 별도."""
    comp = Composition(normalize_user_formula(base_formula))
    atoms = {str(element): float(amount) for element, amount in comp.get_el_amt_dict().items()}
    if source not in atoms or atoms[source] <= 0:
        raise ValueError(f"기본 화학식에 치환할 원소 {source}가 없습니다.")
    if source == target:
        raise ValueError("치환 전후 원소가 같습니다. 다른 원소를 선택하세요.")
    if source not in feature_columns or target not in feature_columns:
        raise ValueError("치환 원소는 현재 모델에 등록된 지원 원소여야 합니다.")
    if len(atoms) == 0 or sum(atoms.values()) <= 0:
        raise ValueError("기본 화학식의 원자 수를 해석할 수 없습니다.")

    base_count = atoms[source]
    total_atoms = sum(atoms.values())
    out = []
    for percent in range(0, 101, 5):
        fraction = percent / 100.0
        current = dict(atoms)
        remain = round(base_count * (1.0 - fraction), 8)
        added = round(base_count * fraction, 8)
        if remain <= 0:
            current.pop(source, None)
        else:
            current[source] = remain
        current[target] = round(current.get(target, 0.0) + added, 8)
        # 조성의 원소 개수 및 원자 수를 0 이상으로 유지한다.
        current = {element: amount for element, amount in current.items() if amount > 0}
        if abs(sum(current.values()) - total_atoms) > 1e-6:
            raise ValueError("원자 수 보존 검사에 실패했습니다.")
        # pymatgen의 조성 표현은 표준 원소 파서를 통해 검증한다.
        def display_amount(value: float) -> str:
            if abs(value - round(value)) < 1e-8:
                return "" if round(value) == 1 else str(int(round(value)))
            return f"{value:.8f}".rstrip("0").rstrip(".")
        formula = "".join(f"{el}{display_amount(amt)}" for el, amt in current.items())
        Composition(formula)
        out.append({
            "치환 비율 (%)": percent,
            "기존 원소 잔존 비율 (%)": 100 - percent,
            "화학식": formula,
        })
    return pd.DataFrame(out)


def v292_substitution_predict(base_formula: str, source: str, target: str,
                              model_v2, v2_columns) -> pd.DataFrame:
    """기존 저장 모델(v1/v2)로만 예측하며 모델을 재학습하지 않습니다."""
    frame = v292_substitution_compositions(base_formula, source, target)
    formulas = frame["화학식"].to_list()
    x1 = pd.concat([make_input(f) for f in formulas], ignore_index=True)
    x2 = pd.concat([make_input_v2(f, v2_columns) for f in formulas], ignore_index=True)
    log1 = np.asarray(model.predict(x1), dtype=float)
    log2 = np.asarray(model_v2.predict(x2), dtype=float)
    if not (np.isfinite(log1).all() and np.isfinite(log2).all()):
        raise ValueError("AI 예측 결과에 유효하지 않은 숫자가 있습니다.")
    frame["v1 log10(S/cm)"] = log1
    frame["v2 log10(S/cm)"] = log2
    frame["v1 예측 이온전도도 (S/cm)"] = np.power(10.0, log1)
    frame["v2 예측 이온전도도 (S/cm)"] = np.power(10.0, log2)
    if not (np.isfinite(frame["v1 예측 이온전도도 (S/cm)"]).all() and
            np.isfinite(frame["v2 예측 이온전도도 (S/cm)"]).all()):
        raise ValueError("예측 이온전도도에 유효하지 않은 수치가 있습니다.")
    return frame


# v29.3: 한 개의 메뉴에서 기존 Cl/Br 및 범용 원소 치환을 모두 실행합니다.
def v293_apply_substitution_preset():
    preset = st.session_state.get("v293_sim_preset", "Cl → Br · PPT 사례")
    defaults = {
        "Cl → Br · PPT 사례": ("Li6PS5Cl", "Cl", "Br"),
        "Ge → Sn · 추가 사례": ("Li10Ge(PS6)2", "Ge", "Sn"),
    }
    if preset in defaults:
        base, old, new = defaults[preset]
        st.session_state["v292_base_formula"] = base
        st.session_state["v292_source_element"] = old
        st.session_state["v292_target_element"] = new
        st.session_state["v292_selected_pct"] = 50
    st.session_state.pop("v292_result", None)


if selected_page == "simulation":
    st.subheader("🧪 조성 변경 시뮬레이션")
    st.write("기본 화학식에서 원소를 바꿔 5% 간격의 가상 조성 21개를 만들고, 두 AI 모델의 전도도 예측 경향을 비교합니다.")
    if "v293_sim_initialized" not in st.session_state:
        st.session_state["v293_sim_preset"] = "Cl → Br · PPT 사례"
        st.session_state["v293_sim_initialized"] = True
        v293_apply_substitution_preset()
    st.radio(
        "예시 선택", ["Cl → Br · PPT 사례", "Ge → Sn · 추가 사례", "직접 입력"],
        horizontal=True, key="v293_sim_preset", on_change=v293_apply_substitution_preset,
    )
    st.caption("원소를 자유롭게 선택하려면 '직접 입력'을 선택하고 아래 화학식과 원소를 변경하세요.")
    with st.expander("시뮬레이션의 한계 · 꼭 알아둘 점", expanded=False):
        st.write("원자 자리를 형식적으로 1:1 치환합니다. 산화수·전하 중성·고용한계·결정상·합성 가능성은 검증하지 않습니다.")
        st.write("예측 최고점은 21개 가상 조성 중 모델 추정이 가장 큰 지점이며, 실험으로 검증된 최적 소재가 아닙니다.")

    if not MODEL_V2_PATH.is_file():
        st.error("두 번째 AI 모델 파일 `SSE_AI_Model_v2.joblib`이 필요합니다.")
    else:
        try:
            gen_model_v2, gen_cols_v2 = load_model_v2()
        except Exception as exc:
            st.error(f"v2 모델 로딩 실패: {exc}")
        else:
            base_formula = st.text_input(
                "기본 고체전해질 화학식", value="Li6PS5Cl",
                help="괄호와 아래첨자를 지원합니다. 예: Li10Ge(PS6)2, Li6PS5Cl",
                key="v292_base_formula",
            )
            try:
                base_comp = Composition(normalize_user_formula(base_formula))
                base_elements = [str(el) for el in base_comp.get_el_amt_dict()]
                missing_base = sorted(set(base_elements) - set(feature_columns))
                if missing_base:
                    raise ValueError("현재 모델에 없는 원소: " + ", ".join(missing_base))
                supported = sorted(set(feature_columns) & _PERIODIC_SYMBOLS)
                if len(supported) < 2:
                    raise ValueError("치환 가능한 학습 원소가 충분하지 않습니다.")
            except Exception as exc:
                st.error(f"기본 화학식을 확인해주세요: {exc}")
            else:
                left, right = st.columns(2)
                with left:
                    if st.session_state.get("v292_source_element") not in base_elements:
                        st.session_state["v292_source_element"] = base_elements[0]
                    source_index = base_elements.index("Cl") if "Cl" in base_elements else 0
                    source = st.selectbox(
                        "기존 원소 (치환할 원소)", base_elements,
                        index=source_index, key="v292_source_element",
                    )
                with right:
                    targets = [el for el in supported if el != source]
                    if st.session_state.get("v292_target_element") not in targets:
                        st.session_state["v292_target_element"] = (
                            "Br" if "Br" in targets else ("Sn" if "Sn" in targets else targets[0])
                        )
                    target_index = targets.index(st.session_state["v292_target_element"])
                    target = st.selectbox(
                        "새 원소 (대체할 원소)", targets,
                        index=target_index, key="v292_target_element",
                    )
                if "Sn" not in supported and source == "Ge":
                    st.caption("현재 모델의 지원 원소에 Sn이 없으므로 Sn 대신 지원되는 원소만 선택할 수 있습니다.")
                selected_pct = st.slider(
                    "보여줄 치환 비율 (%)", min_value=0, max_value=100,
                    value=50, step=5, key="v292_selected_pct",
                )
                current_signature = (normalize_user_formula(base_formula), source, target)
                try:
                    preview = v292_substitution_compositions(base_formula, source, target)
                    preview_formula = preview.loc[
                        preview["치환 비율 (%)"] == selected_pct, "화학식"
                    ].iloc[0]
                    st.code(preview_formula, language="text")
                    st.caption(
                        f"{source}의 원래 원자 자리 중 {selected_pct}%를 {target}로 바꾼 "
                        "가상 조성입니다. 모델 입력으로 쓰이는 원소 비율만 변경됩니다."
                    )
                except Exception as exc:
                    st.error(f"조성 생성 실패: {exc}")

                if st.button("🔬 21개 가상 조성 예측하기", use_container_width=True, key="v292_run"):
                    try:
                        with st.spinner("21개 조성을 기존 AI 두 모델로 예측하고 있습니다..."):
                            result = v292_substitution_predict(
                                base_formula, source, target, gen_model_v2, gen_cols_v2
                            )
                        st.session_state["v292_result"] = {
                            "signature": current_signature, "frame": result,
                        }
                    except Exception as exc:
                        st.session_state.pop("v292_result", None)
                        st.error(f"범용 조성 예측 실패: {exc}")

                stored = st.session_state.get("v292_result")
                if stored is not None and stored["signature"] != current_signature:
                    st.info("화학식이나 치환 원소를 변경했습니다. 버튼을 다시 눌러 새로운 결과를 계산하세요.")
                elif stored is not None:
                    data = stored["frame"]
                    point = data.loc[data["치환 비율 (%)"] == selected_pct].iloc[0]
                    st.markdown("#### 선택한 조성의 AI 예측값")
                    a, b = st.columns(2)
                    a.metric("v1 · 52개 특징 (S/cm)", f"{point['v1 예측 이온전도도 (S/cm)']:.4e}")
                    b.metric("v2 · 59개 특징 (S/cm)", f"{point['v2 예측 이온전도도 (S/cm)']:.4e}")
                    st.caption(f"선택된 가상 조성: {point['화학식']}")

                    graph = data.melt(
                        id_vars=["치환 비율 (%)", "화학식"],
                        value_vars=["v1 log10(S/cm)", "v2 log10(S/cm)"],
                        var_name="AI 모델", value_name="예측 log₁₀(S/cm)",
                    )
                    fig = px.line(
                        graph, x="치환 비율 (%)", y="예측 log₁₀(S/cm)",
                        color="AI 모델", markers=True, custom_data=["화학식"],
                        title=f"{source} → {target} 치환에 따른 AI 예측 이온전도도 변화",
                    )
                    fig.add_vline(x=selected_pct, line_dash="dot", line_color="gray")
                    fig.update_traces(
                        hovertemplate="치환 비율: %{x:.0f}%<br>예측 log₁₀: %{y:.3f}<br>화학식: %{customdata[0]}<extra>%{fullData.name}</extra>"
                    )
                    fig.update_layout(height=410, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("두 곡선은 모델이 예측한 값이며 실험 측정 곡선이 아닙니다. 로그값 1 차이는 전도도 10배에 해당합니다.")
                    a, b = st.columns(2)
                    for col, model_name, axis in [
                        (a, "v1", "v1 예측 이온전도도 (S/cm)"),
                        (b, "v2", "v2 예측 이온전도도 (S/cm)"),
                    ]:
                        top = data.loc[data[axis].idxmax()]
                        col.metric(f"{model_name} · 전체 21개 중 최고 예측 비율", f"{int(top['치환 비율 (%)'])}%")
                        col.caption(f"{top['화학식']} · {top[axis]:.3e} S/cm")
                    with st.expander("전체 21개 가상 조성·CSV 다운로드", expanded=False):
                        st.markdown("#### 전체 21개 가상 조성 비교표")
                        view = data.copy()
                        for axis in ["v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"]:
                            view[axis] = view[axis].map(lambda value: f"{value:.4e}")
                        v29_show_table(
                            view,
                            ["치환 비율 (%)", "화학식", "v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"],
                            key="v292_result_table",
                        )
                        st.download_button(
                            "📥 범용 원소 치환 예측 결과 CSV 다운로드",
                            data=data.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"SSE_AI_{APP_FILE_TAG}_generic_substitution.csv",
                            mime="text/csv", key="v292_download",
                        )
                else:
                    st.info("위 버튼을 누르면 21개 조성의 AI 예측 그래프와 비교표가 생성됩니다.")

# ─────────────────────────────────────────────────────────────────────────────
# 여덟 번째 메뉴 (v9): 예측값 해석과 학습 조성 유사도 / 모델 간 차이
# ─────────────────────────────────────────────────────────────────────────────
def v9_disagreement(log1: float, log2: float) -> tuple[float, float]:
    """두 모델의 log10 예측 차이와 이온전도도 비율 (큰 값/작은 값)."""
    gap_log = abs(float(log1) - float(log2))
    if not np.isfinite(gap_log):
        raise ValueError("두 모델의 로그 예측값이 유한한 숫자가 아닙니다.")
    return gap_log, float(10.0 ** gap_log)


def v9_nearest_compositions(formulas: list[str], train_meta: pd.DataFrame,
                            train_vectors: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """v1에서 사용하는 52개 원소 원자분율 기반 TV 거리로 가장 가까운 학습 조성을 찾음.

    유사도(%) = 100 * (1 - 0.5 * L1_distance). 통계적 예측 확률이 아님.
    학습 참조 CSV만 사용하고 공식 테스트 데이터는 사용하지 않음.
    """
    if not len(formulas) or train_vectors.empty:
        raise ValueError("비교할 화학식이나 학습 참조 조성이 없습니다.")
    if len(train_vectors) != len(train_meta):
        raise ValueError("학습 참조 정보와 특징 행 수가 일치하지 않습니다.")

    query_vectors = np.vstack([make_input(s).iloc[0].to_numpy(dtype=float) for s in formulas])
    reference_vectors = train_vectors.loc[:, feature_columns].to_numpy(dtype=float)
    if query_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError("입력 조성 특징과 학습 참조 특징의 열 수가 다릅니다.")

    # [입력 조성 개수, 학습 소재 수], 복잡한 결정구조 정보는 계산하지 않음.
    distances = 0.5 * np.abs(query_vectors[:, None, :] - reference_vectors[None, :, :]).sum(axis=2)
    nearest_idx = distances.argmin(axis=1)
    best_distances = distances[np.arange(len(formulas)), nearest_idx]
    similarity = 100.0 * (1.0 - np.clip(best_distances, 0, 1))
    exact_counts = np.isclose(distances, 0.0, atol=1e-10, rtol=0).sum(axis=1).astype(int)
    top_nearest = train_meta.iloc[nearest_idx]["화학식"].astype(str).to_numpy()

    report = pd.DataFrame({
        "화학식": formulas,
        "가장 가까운 학습 조성": top_nearest,
        "학습 조성 유사도 (%)": similarity,
        "동일 조성 학습 기록 (건)": exact_counts,
    })
    return report, distances


def v9_warning_messages(exact_count: int | None, nearest_similarity: float | None,
                        ratio: float) -> list[str]:
    """검증된 신뢰도 등급이 아닌, 관찰 가능한 사실만 설명하는 주의 문장."""
    notes = []
    if exact_count is None:
        notes.append("학습 참조 CSV가 없어 실제 학습 조성과의 비교를 수행하지 못했습니다.")
    elif exact_count:
        notes.append(
            f"같은 원소 조성비의 학습 기록이 {exact_count}건 있습니다. "
            "이는 새로운 조성에 대한 독립 검증이나 모델 예측 정확도의 근거가 아닙니다."
        )
    else:
        notes.append(
            f"완전히 같은 학습 조성은 없습니다. 가장 가까운 조성 유사도는 {nearest_similarity:.1f}%이며, "
            "이 비율은 예측 정확도나 신뢰도 확률이 아닙니다."
        )
    if ratio >= 2:
        notes.append(
            f"두 AI 모델의 예측 이온전도도가 약 {ratio:.2f}배 차이납니다. "
            "이 차이는 검증된 예측 불확실성 구간이 아니며, 어느 모델이 더 정확한지도 알 수 없습니다."
        )
    else:
        notes.append(
            f"두 AI 모델의 예측 차이는 약 {ratio:.2f}배입니다. "
            "예측값이 비슷하더라도 실제 측정값에 가깝다는 보장은 없습니다."
        )
    notes.append(
        "현재 모델에는 결정구조·합성법·온도·계면 안정성이 직접 포함되지 않습니다. "
        "실제 적용 전에 문헌과 실험으로 확인해야 합니다."
    )
    return notes


if selected_page == "interpretation":
    st.subheader("🧭 v9 · AI 예측값 해석과 주의 신호")
    st.write(
        "**① 실제 학습 자료와의 원소 조성비 유사도, ② 같은 조성의 학습 기록 수, "
        "③ v1·v2 모델 예측값 차이**를 보여줍니다."
    )
    st.info(
        "여기 표시하는 수치는 **모델의 정확도 %, 신뢰도 확률 또는 보정된 예측구간**이 아닙니다. "
        "유사도는 52개 원소 조성비 사이의 거리이며, 두 모델의 차이는 단지 서로 다른 계산 결과입니다. "
        "OBELiX 공식 테스트 데이터는 이 기능에서 불러오지 않습니다."
    )

    if not MODEL_V2_PATH.is_file():
        st.error("v2 모델 파일(`SSE_AI_Model_v2.joblib`)이 없습니다. 기존 앱 폴더에 보관된 파일을 확인하세요.")
    else:
        try:
            v9_model_v2, v9_v2_columns = load_model_v2()
        except Exception as exc:
            st.error("v2 모델을 불러올 수 없습니다. 모델 파일과 scikit-learn 버전을 확인하세요.")
            st.code(str(exc))
        else:
            st.markdown("#### 1. 화학식 하나의 예측 결과 분석")
            with st.form("v9_interpretation_form"):
                inspect_formula = st.text_input(
                    "해석할 고체전해질 화학식", value="Li6PS5Cl0.5Br0.5",
                    help="예시: Li6PS5Cl, Li6PS5Cl0.5Br0.5, Li6PS5Br, Li3PS4"
                )
                inspect_click = st.form_submit_button("🔍 예측값과 학습 조성 분석", use_container_width=True)

            if inspect_click:
                try:
                    cleaned = normalize_user_formula(inspect_formula)
                    v9_in1 = make_input(cleaned)
                    v9_in2 = make_input_v2(cleaned, v9_v2_columns)
                    pred1 = float(model.predict(v9_in1)[0])
                    pred2 = float(v9_model_v2.predict(v9_in2)[0])
                    gap_log, ratio = v9_disagreement(pred1, pred2)
                    analysis_record = {
                        "화학식": cleaned,
                        "v1_log": pred1,
                        "v2_log": pred2,
                        "예측 차이 (log10)": gap_log,
                        "두 모델 예측 비율 (배)": ratio,
                        "학습 참조 파일 상태": "없음",
                        "유사도": None,
                        "동일 기록 수": None,
                    }
                    if TRAIN_REFERENCE_PATH.is_file():
                        train_meta, train_vectors, total, skipped = get_training_reference()
                        neighbors, distances = v9_nearest_compositions(
                            [cleaned], train_meta, train_vectors
                        )
                        near = neighbors.iloc[0]
                        analysis_record.update({
                            "유사도": float(near["학습 조성 유사도 (%)"]),
                            "동일 기록 수": int(near["동일 조성 학습 기록 (건)"]),
                            "가장 유사한 학습 소재": str(near["가장 가까운 학습 조성"]),
                            "학습 참조 파일 상태": f"{total}개 행 (분석 {len(train_meta)}개)",
                        })
                        dist = distances[0]
                        ordered = np.argsort(dist)
                        lookup = train_meta.iloc[ordered].copy()
                        lookup["원소 조성 유사도 (%)"] = 100 * (1 - np.clip(dist[ordered], 0, 1))
                        # 같은 조성이 여러 건 있더라도 설명용 이웃 목록은 조성별 하나만 보여줌.
                        lookup = lookup.drop_duplicates("표준 화학식").head(5)
                        analysis_record["유사 소재 표"] = lookup[[
                            "화학식", "소재 계열", "원소 조성 유사도 (%)", "학습 측정값 (S/cm)"
                        ]]
                    st.session_state["v9_single_analysis"] = analysis_record
                except Exception as exc:
                    st.session_state.pop("v9_single_analysis", None)
                    st.error("화학식 분석을 완료하지 못했습니다.")
                    st.code(str(exc))

            if "v9_single_analysis" in st.session_state:
                current = st.session_state["v9_single_analysis"]
                st.markdown(f"**분석 대상:** `{current['화학식']}`")
                a, b, c = st.columns(3)
                a.metric("v1 예측 (S/cm)", f"{10 ** current['v1_log']:.3e}")
                b.metric("v2 예측 (S/cm)", f"{10 ** current['v2_log']:.3e}")
                c.metric("두 모델 예측 비율 (큰 값/작은 값)", f"{current['두 모델 예측 비율 (배)']:.2f}배")
                st.caption(
                    f"모델 간 차이: {current['예측 차이 (log10)']:.3f} log₁₀ 단위. "
                    "이 값은 예측 오차나 검증된 불확실성이 아닙니다."
                )
                if current["유사도"] is None:
                    st.warning(
                        "`SSE_train_reference.csv`가 없어 학습 소재와의 유사도를 계산하지 못했습니다. "
                        "같은 폴더에 기존 학습 참조 CSV를 놓으면 분석할 수 있습니다."
                    )
                else:
                    p, q = st.columns(2)
                    p.metric("가장 가까운 학습 조성 유사도", f"{current['유사도']:.1f}%")
                    q.metric("완전히 같은 학습 조성 기록", f"{current['동일 기록 수']}건")
                    st.caption(
                        f"가장 가까운 학습 소재: {current['가장 유사한 학습 소재']} · "
                        f"Colab 학습 참조 {current['학습 참조 파일 상태']}"
                    )
                    st.markdown("**가장 가까운 학습 소재 5종 (조성별 중복 제거)**")
                    near_view = current["유사 소재 표"].copy()
                    near_view["원소 조성 유사도 (%)"] = near_view["원소 조성 유사도 (%)"].map(
                        lambda v: f"{v:.1f}"
                    )
                    st.dataframe(near_view, use_container_width=True, hide_index=True)

                st.markdown("**현재 예측을 해석할 때 확인할 점**")
                for item in v9_warning_messages(
                    current["동일 기록 수"], current["유사도"], current["두 모델 예측 비율 (배)"]
                ):
                    st.write("• " + item)

            st.divider()
            st.markdown("#### 2. Cl/Br 가상 조성 21개를 함께 점검")
            st.write(
                "v8에서 시뮬레이션한 **21개 조성 전체**를 대상으로 모델 간 차이를 계산합니다. "
                "학습 참조 파일이 있으면 가장 가까운 학습 소재와 유사도도 함께 표시합니다."
            )
            if st.button("📊 21개 조성의 주의 신호 분석", use_container_width=True, key="v9_check_sweep"):
                try:
                    # v8에서 계산한 값을 재사용하여 두 탭에서 일관된 결과를 표시한다.
                    raw_sweep = st.session_state.get("v8_sweep_results")
                    if raw_sweep is None:
                        raw_sweep = run_halide_sweep(v9_model_v2, v9_v2_columns)
                    sweep_report = raw_sweep.copy()
                    log_gap = (
                        sweep_report["v1 log10(S/cm)"] - sweep_report["v2 log10(S/cm)"]
                    ).abs()
                    sweep_report["모델 간 로그 예측 차이"] = log_gap
                    sweep_report["두 모델 예측 비율 (배)"] = np.power(10.0, log_gap)
                    if TRAIN_REFERENCE_PATH.is_file():
                        train_meta, train_vectors, total, skipped = get_training_reference()
                        nearest, _ = v9_nearest_compositions(
                            sweep_report["화학식"].astype(str).to_list(), train_meta, train_vectors
                        )
                        for col in ["가장 가까운 학습 조성", "학습 조성 유사도 (%)", "동일 조성 학습 기록 (건)"]:
                            sweep_report[col] = nearest[col].to_numpy()
                    st.session_state["v9_sweep_report"] = sweep_report
                except Exception as exc:
                    st.session_state.pop("v9_sweep_report", None)
                    st.error("21개 조성 분석을 완료하지 못했습니다.")
                    st.code(str(exc))

            if "v9_sweep_report" in st.session_state:
                sweep_report = st.session_state["v9_sweep_report"]
                similar_col = "학습 조성 유사도 (%)"
                if similar_col in sweep_report.columns:
                    left, right = st.columns(2)
                    with left:
                        chart_sim = px.line(
                            sweep_report, x="Br 치환 비율 (%)", y=similar_col,
                            markers=True, title="Br 비율에 따른 가장 가까운 학습 조성 유사도",
                            labels={similar_col: "학습 조성 유사도 (%)"}
                        )
                        valid_sim = pd.to_numeric(sweep_report[similar_col], errors="coerce").dropna()
                        if not valid_sim.empty:
                            low_sim, high_sim = float(valid_sim.min()), float(valid_sim.max())
                            padding_sim = max(0.15, (high_sim - low_sim) * 0.28)
                            chart_sim.update_yaxes(
                                range=[max(0.0, low_sim - padding_sim), min(100.3, high_sim + padding_sim)],
                                ticksuffix="%",
                            )
                        chart_sim.update_layout(height=360)
                        st.plotly_chart(chart_sim, use_container_width=True)
                        st.caption("원소 조성비 유사도가 100% 근처이므로 세로축을 확대했습니다. 정확도 그래프가 아닙니다.")
                    with right:
                        chart_gap = px.line(
                            sweep_report, x="Br 치환 비율 (%)", y="모델 간 로그 예측 차이",
                            markers=True, title="Br 비율에 따른 두 AI 모델의 차이"
                        )
                        chart_gap.update_yaxes(rangemode="tozero")
                        chart_gap.update_layout(height=360)
                        st.plotly_chart(chart_gap, use_container_width=True)
                else:
                    st.warning(
                        "학습 참조 CSV가 없어 학습 조성 유사도를 표시하지 못했습니다. "
                        "두 모델의 예측 차이만 계산했습니다."
                    )
                    chart_gap = px.line(
                        sweep_report, x="Br 치환 비율 (%)", y="모델 간 로그 예측 차이",
                        markers=True, title="Br 비율에 따른 두 AI 모델의 차이"
                    )
                    st.plotly_chart(chart_gap, use_container_width=True)
                st.caption(
                    "유사도 100%는 해당 조성의 학습 기록이 있다는 뜻이며 정확도 100%가 아닙니다. "
                    "모델 차이가 작다고 예측이 검증됐다는 뜻도 아닙니다."
                )
                display_cols = [
                    "Br 치환 비율 (%)", "화학식", "v1 예측 이온전도도 (S/cm)",
                    "v2 예측 이온전도도 (S/cm)", "두 모델 예측 비율 (배)"
                ]
                if similar_col in sweep_report.columns:
                    display_cols += [similar_col, "동일 조성 학습 기록 (건)", "가장 가까운 학습 조성"]
                show = sweep_report[display_cols].copy()
                for col in ["v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"]:
                    show[col] = show[col].map(lambda x: f"{x:.3e}")
                show["두 모델 예측 비율 (배)"] = show["두 모델 예측 비율 (배)"].map(lambda x: f"{x:.2f}")
                if similar_col in show.columns:
                    show[similar_col] = show[similar_col].map(lambda x: f"{x:.1f}")
                v29_show_table(show,
                    ["Br 치환 비율 (%)", "화학식", "v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)", "두 모델 예측 비율 (배)"],
                    key="v29_interpret_table")
                st.download_button(
                    "📥 21개 조성 해석 결과 CSV 저장",
                    data=sweep_report.to_csv(index=False).encode("utf-8-sig"),
                    file_name="SSE_AI_v9_prediction_interpretation.csv", mime="text/csv",
                    key="v9_report_download"
                )
            else:
                st.caption("버튼을 누르면 21개 조성의 학습 데이터 유사도와 모델 간 차이를 표시합니다.")

            st.divider()
            st.markdown("#### 3. 모델 평가 결과와 해석상의 한계")
            st.write(
                "기존 v1 모델의 **OBELiX 공식 테스트 109건**에서 MAE(로그₁₀)는 **1.522**, "
                "R²는 **−0.084**로 기록되어 있습니다. v2의 공식 테스트 성능은 아직 검증되지 않았습니다. "
                "따라서 원소 조성비가 비슷하거나 두 모델이 일치하더라도 실험값이 맞는다고 판단해서는 안 됩니다."
            )
            st.caption(
                "위 수치는 기존 평가 기록입니다. 이 v9 화면에서는 공식 테스트를 재실행하지 않으며, "
                "그 테스트 결과로 모델을 다시 최적화하지 않습니다."
            )



# ─────────────────────────────────────────────────────────────────────────────
# 아홉 번째 메뉴 (v10): 조성별 결과를 통합하고 한국어 PDF 분석 보고서로 내보내기
# 기존 v8/v9 모델과 함수만 재사용하며, 공식 테스트 데이터를 불러오지 않는다.
# PDF 생성에만 reportlab 라이브러리가 필요하다 (pip install reportlab).
# ─────────────────────────────────────────────────────────────────────────────

def v10_create_dashboard_data(model_v2, v2_columns):
    """21개 조성 예측을 모으고 (가능하면) 학습 참조 데이터와 연결한다."""
    original = st.session_state.get("v8_sweep_results")
    if original is None:
        original = run_halide_sweep(model_v2, v2_columns)
    combined = original.copy().reset_index(drop=True)
    gap = (combined["v1 log10(S/cm)"] - combined["v2 log10(S/cm)"]).abs()
    combined["모델 간 로그 예측 차이"] = gap
    combined["두 모델 예측 비율 (배)"] = np.power(10.0, gap)

    train_meta, distance_matrix, training_note = None, None, None
    if TRAIN_REFERENCE_PATH.is_file():
        try:
            meta, vectors, raw_count, excluded = get_training_reference()
            nearest, dists = v9_nearest_compositions(
                combined["화학식"].astype(str).to_list(), meta, vectors
            )
            for col in ["가장 가까운 학습 조성", "학습 조성 유사도 (%)", "동일 조성 학습 기록 (건)"]:
                combined[col] = nearest[col].to_numpy()
            train_meta = meta
            distance_matrix = dists
            training_note = f"학습 참조 {raw_count}건 중 유효 {len(meta)}건"
            if excluded:
                training_note += f" (파싱 제외 {excluded}건)"
        except Exception as exc:
            training_note = f"학습 참조 자료를 읽지 못했습니다: {exc}"
    else:
        training_note = "SSE_train_reference.csv 없음: 조성 유사도와 학습 측정값은 생략됨"
    return {
        "sweep": combined,
        "train_meta": train_meta,
        "distances": distance_matrix,
        "training_note": training_note,
    }


def v10_selected_detail(report_data: dict, br_percent: int) -> dict:
    """슬라이더에서 선택한 조성의 수치 및 정확히 같은 학습 기록을 반환한다."""
    sweep = report_data["sweep"]
    selection = np.where(sweep["Br 치환 비율 (%)"].to_numpy() == br_percent)[0]
    if not len(selection):
        raise ValueError("선택한 치환 비율에 대한 결과가 없습니다.")
    row_index = int(selection[0])
    row = sweep.iloc[row_index]
    matches = pd.DataFrame()
    if report_data["train_meta"] is not None and report_data["distances"] is not None:
        exact_mask = np.isclose(report_data["distances"][row_index], 0, rtol=0, atol=1e-10)
        matches = report_data["train_meta"].loc[exact_mask].copy()
    return {"point": row, "exact_matches": matches}


def v10_register_korean_pdf_font() -> str:
    """사용자 PC의 한국어 글꼴을 PDF에 포함한다. 글꼴 파일은 배포하지 않는다."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from pathlib import Path

    font_name = "SSEv10Korean"
    if font_name in pdfmetrics.getRegisteredFontNames():
        return font_name
    # Windows 기본 맑은 고딕을 우선 사용, 다른 OS는 사용 가능한 한글 글꼴 탐색.
    paths = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"),
    ]
    for path in paths:
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(path)))
                return font_name
            except Exception:
                continue
    # ReportLab 내장 한글 CID 글꼴 (OS 환경별 PDF 표시 상태가 다를 수 있음).
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    return "HYSMyeongJo-Medium"


def v10_build_pdf(report_data: dict, detail: dict) -> bytes:
    """현재 화면과 동일한 수치로 표·곡선을 포함한 한국어 2페이지 PDF 생성."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
    from reportlab.graphics.shapes import Drawing, Line, Rect, String, PolyLine, Circle

    pdf_font = v10_register_korean_pdf_font()
    navy, teal, blue = HexColor("#11233B"), HexColor("#078C85"), HexColor("#3C71CD")
    text_grey, pale, border = HexColor("#5A6678"), HexColor("#F0F6F6"), HexColor("#DCE5EA")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=43, rightMargin=43, topMargin=43, bottomMargin=42,
        title=f"SSE-AI Lab {APP_VERSION} - 조성 예측 분석 보고서",
        author="SSE-AI Lab",
    )
    width = A4[0] - 86
    styles = {
        "eyebrow": ParagraphStyle("eyebrow", fontName=pdf_font, fontSize=9, leading=14, textColor=teal, spaceAfter=9, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName=pdf_font, fontSize=20, leading=28, textColor=navy, spaceAfter=9, wordWrap="CJK"),
        "section": ParagraphStyle("section", fontName=pdf_font, fontSize=12, leading=19, textColor=navy, spaceBefore=11, spaceAfter=8, wordWrap="CJK"),
        "body": ParagraphStyle("body", fontName=pdf_font, fontSize=9.3, leading=15.5, textColor=navy, spaceAfter=5, wordWrap="CJK"),
        "small": ParagraphStyle("small", fontName=pdf_font, fontSize=8, leading=12, textColor=text_grey, spaceAfter=5, wordWrap="CJK"),
        "table": ParagraphStyle("table", fontName=pdf_font, fontSize=8.1, leading=11, textColor=navy, wordWrap="CJK"),
        "th": ParagraphStyle("th", fontName=pdf_font, fontSize=8.0, leading=11, textColor=colors.white, wordWrap="CJK"),
    }
    P = lambda v, kind="body": Paragraph(escape(str(v)), styles[kind])
    row = detail["point"]
    sim = row.get("학습 조성 유사도 (%)", np.nan)
    count = row.get("동일 조성 학습 기록 (건)", np.nan)
    v1, v2 = float(row["v1 예측 이온전도도 (S/cm)"]), float(row["v2 예측 이온전도도 (S/cm)"])
    ratio = float(row["두 모델 예측 비율 (배)"])
    br = int(row["Br 치환 비율 (%)"])
    formula = str(row["화학식"])
    exact = detail["exact_matches"]
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M KST")

    story = [P(f"SSE-AI LAB / {APP_VERSION.upper()} REPORT", "eyebrow"),
             P("고체전해질 조성 예측 분석 보고서", "title"),
             P(f"생성 시각: {stamp}  |  조성: {formula}  |  Br 치환 비율: {br}%", "small"),
             Spacer(1, 10), P("01  선택한 조성의 AI 예측 결과", "section")]
    metrics = [
        [P("v1 예측 (S/cm)", "small"), P("v2 예측 (S/cm)", "small")],
        [P(f"{v1:.4e}", "section"), P(f"{v2:.4e}", "section")],
        [P("두 모델 예측 비율", "small"), P("가장 가까운 학습 조성 유사도", "small")],
        [P(f"{ratio:.2f}배", "section"), P(f"{float(sim):.1f}%" if pd.notna(sim) else "자료 없음", "section")],
    ]
    mt = Table(metrics, colWidths=[width/2] * 2, rowHeights=[21, 33, 21, 33])
    mt.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),0.6,border),
        ("INNERGRID",(0,0),(-1,-1),0.5,border),
        ("BACKGROUND",(0,0),(-1,-1),pale),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),13),
    ]))
    story += [mt, Spacer(1, 8)]

    if pd.notna(count):
        story.append(P(f"동일 조성 학습 기록: {int(count)}건. 가장 유사한 학습 소재: {row.get('가장 가까운 학습 조성', '없음')}", "body"))
    else:
        story.append(P("학습 참조 CSV가 없어 동일 조성 및 유사도 정보는 생략되었습니다.", "body"))
    if exact.empty:
        story.append(P("선택 조성에 대한 정확히 일치하는 학습 측정값은 확인되지 않았습니다.", "small"))
    else:
        values = pd.to_numeric(exact["학습 측정값 (S/cm)"], errors="coerce")
        valid = values[np.isfinite(values) & (values > 0)]
        if len(valid):
            shown_values = ", ".join(f"{n:.3e}" for n in valid.iloc[:5])
            tail = f" 외 {len(valid)-5}건" if len(valid) > 5 else ""
            story.append(P(f"동일 조성의 학습 측정값 ({len(valid)}건, S/cm): {shown_values}{tail}", "body"))
            story.append(P("주의: 위 값은 모델 학습에 사용된 자료이며 독립적인 정확도 검증 결과가 아닙니다.", "small"))

    story.append(P("02  Cl/Br 치환 비율별 예측 변화", "section"))
    sweep = report_data["sweep"]
    w, h = width, 204
    drawing = Drawing(w, h)
    x0, x1, y0, y1 = 55, w-19, 35, 159
    s1 = sweep["v1 log10(S/cm)"].to_numpy(dtype=float)
    s2 = sweep["v2 log10(S/cm)"].to_numpy(dtype=float)
    brs = sweep["Br 치환 비율 (%)"].to_numpy(dtype=float)
    y_min = float(min(s1.min(), s2.min()) - 0.12)
    y_max = float(max(s1.max(), s2.max()) + 0.12)
    xcoord = lambda n: x0 + (x1-x0)*float(n)/100
    ycoord = lambda n: y0 + (y1-y0)*(float(n)-y_min)/(y_max-y_min)
    drawing.add(Rect(0,0,w,h,fillColor=colors.white,strokeColor=border,strokeWidth=0.8))
    for t in np.linspace(y_min,y_max,5):
        yy = ycoord(t)
        drawing.add(Line(x0,yy,x1,yy,strokeColor=border,strokeWidth=0.6))
        drawing.add(String(x0-8,yy-3,f"{t:.2f}",fontName=pdf_font,fontSize=7,textAnchor="end",fillColor=text_grey))
    for p in range(0,101,20):
        xx = xcoord(p)
        drawing.add(String(xx,y0-13,str(p),fontName=pdf_font,fontSize=7,textAnchor="middle",fillColor=text_grey))
    drawing.add(Line(xcoord(br),y0,xcoord(br),y1,strokeColor=teal,strokeWidth=1.2,strokeDashArray=[3,3]))
    for series, color in [(s1,blue),(s2,teal)]:
        points = []
        for x,y in zip(brs,series):
            points.extend([xcoord(x),ycoord(y)])
        drawing.add(PolyLine(points,strokeColor=color,strokeWidth=1.8))
        for i in [0,5,10,15,20]:
            drawing.add(Circle(xcoord(brs[i]),ycoord(series[i]),2.3,fillColor=color,strokeColor=color))
        selected_idx = int(np.argmin(np.abs(brs-br)))
        drawing.add(Circle(xcoord(br),ycoord(series[selected_idx]),3.8,fillColor=colors.white,strokeColor=color,strokeWidth=1.7))
    drawing.add(String((x0+x1)/2,13,"Br 치환 비율 (%)",fontName=pdf_font,fontSize=8,textAnchor="middle",fillColor=navy))
    drawing.add(Line(w-171,h-14,w-150,h-14,strokeColor=blue,strokeWidth=2))
    drawing.add(String(w-145,h-17,"v1 (52개 특징)",fontName=pdf_font,fontSize=8,fillColor=navy))
    drawing.add(Line(w-84,h-14,w-65,h-14,strokeColor=teal,strokeWidth=2))
    drawing.add(String(w-61,h-17,"v2 (59개)",fontName=pdf_font,fontSize=8,fillColor=navy))
    story += [drawing,
              Spacer(1, 8),
              P("두 곡선은 5% 간격 21개 가상 조성의 예측 log10(S/cm)를 연결한 것으로, 실험 측정 곡선이 아닙니다.", "small"),
              P("03  결과 해석 시 주의사항", "section"),
              P("- 조성 유사도는 원소 원자분율의 유사성이지 예측 정확도나 신뢰도 확률이 아닙니다.", "small"),
              P("- 모델 간 예측 차이가 작더라도 실제 이온전도도와 일치한다고 보장할 수 없습니다.", "small"),
              P("- v1 공식 테스트 기록: 109건, MAE(log10) 1.522, R² -0.084. v2 공식 테스트 성능은 미평가입니다.", "small"),
              PageBreak(),
              P("SSE-AI LAB / 21 COMPOSITIONS", "eyebrow"),
              P("21개 가상 조성 예측 결과표", "title"),
              P("Li6PS5Cl(1-x)Br(x)에서 x=0~1을 0.05 간격으로 설정한 모델 추정값", "small"),
              Spacer(1, 12)]

    table_headers = ["Br (%)", "화학식", "v1 (S/cm)", "v2 (S/cm)", "비율(배)", "유사도(%)", "학습 건수"]
    grid = [[P(head,"th") for head in table_headers]]
    highlight_index = None
    for i, item in sweep.iterrows():
        numeric_sim = item.get("학습 조성 유사도 (%)", np.nan)
        n_exact = item.get("동일 조성 학습 기록 (건)", np.nan)
        cells = [
            str(int(item["Br 치환 비율 (%)"])),
            str(item["화학식"]),
            f"{float(item['v1 예측 이온전도도 (S/cm)']):.3e}",
            f"{float(item['v2 예측 이온전도도 (S/cm)']):.3e}",
            f"{float(item['두 모델 예측 비율 (배)']):.2f}",
            f"{float(numeric_sim):.1f}" if pd.notna(numeric_sim) else "-",
            str(int(n_exact)) if pd.notna(n_exact) else "-",
        ]
        grid.append([P(c,"table") for c in cells])
        if int(item["Br 치환 비율 (%)"]) == br:
            highlight_index = i + 1
    table = Table(grid, colWidths=[48,125,72,72,63,67,62], rowHeights=[28]+[21]*21, repeatRows=1)
    ts = [
        ("BACKGROUND",(0,0),(-1,0),navy),
        ("BOX",(0,0),(-1,-1),0.5,border),
        ("INNERGRID",(0,0),(-1,-1),0.35,border),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),2),
    ]
    for r in range(2,len(grid),2):
        ts.append(("BACKGROUND",(0,r),(-1,r),HexColor("#F7FAFC")))
    if highlight_index is not None:
        ts.append(("BACKGROUND",(0,highlight_index),(-1,highlight_index),HexColor("#DFF4F0")))
    table.setStyle(TableStyle(ts))
    story.extend([table,Spacer(1,15),
                  P("유사도 = 100 × (1 - 0.5 × 원소 조성비 L1 거리). 이 수치는 예측의 정확도, 신뢰구간, 실험 검증을 의미하지 않습니다.", "small"),
                  P(f"데이터: OBELiX 공개 학습 자료(참조 파일이 있을 경우) | {report_data['training_note']}", "small"),
                  P("이 보고서는 연구 후보 탐색용으로 생성되었습니다. 합성 가능성, 결정구조, 측정 조건, 안정성은 별도로 검토해야 합니다.", "small")])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(border)
        canvas.line(43,32,A4[0]-43,32)
        canvas.setFont(pdf_font,7.2)
        canvas.setFillColor(text_grey)
        canvas.drawString(43,21,f"SSE-AI Lab {APP_VERSION} | AI 예측 결과 · 비검증 참고자료")
        canvas.drawRightString(A4[0]-43,21,str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


if selected_page == "dashboard":
    st.subheader("📑 통합 소재 분석 대시보드 및 PDF 보고서")
    st.write(
        "Cl/Br 치환 조성의 **두 모델 예측**, **가장 가까운 실제 학습 소재**, "
        "**학습 측정값(있는 경우)**, **21개 변화 그래프**를 모아 확인하고 PDF로 저장합니다."
    )
    st.info(
        "PDF에 들어가는 수치는 AI의 추정값입니다. 학습 측정값은 실험 기록이지만 모델 학습에 사용된 "
        "자료이므로 독립적인 정확도 검증이 아닙니다. 공식 테스트 데이터는 다시 불러오지 않습니다."
    )

    if not MODEL_V2_PATH.is_file():
        st.error("두 번째 AI 모델 파일 `SSE_AI_Model_v2.joblib`을 앱과 같은 폴더에 넣어주세요.")
    else:
        try:
            v10_model_v2, v10_columns_v2 = load_model_v2()
        except Exception as exc:
            st.error("두 번째 AI 모델을 불러오지 못했습니다.")
            st.code(str(exc))
        else:
            chosen_pct = st.slider(
                "보고서로 만들 Br 치환 비율 (%)", min_value=0, max_value=100,
                value=50, step=5, key="v10_br_pct"
            )
            st.caption(f"선택한 화학식: `{make_halide_formula(chosen_pct/100):s}`")
            if st.button("📊 통합 분석 실행", use_container_width=True, key="v10_run"):
                try:
                    with st.spinner("21개 조성을 분석하고 학습 자료를 확인하는 중..."):
                        st.session_state["v10_dashboard_data"] = v10_create_dashboard_data(
                            v10_model_v2, v10_columns_v2
                        )
                    st.success("통합 분석을 완료했습니다. 슬라이더를 움직이면 선택 조성만 바뀝니다.")
                except Exception as exc:
                    st.session_state.pop("v10_dashboard_data", None)
                    st.error("통합 분석을 완료하지 못했습니다. AI 모델 파일을 확인해주세요.")
                    st.code(str(exc))

            if "v10_dashboard_data" in st.session_state:
                packed = st.session_state["v10_dashboard_data"]
                try:
                    detail_v10 = v10_selected_detail(packed,chosen_pct)
                    current_v10 = detail_v10["point"]
                    exact_v10 = detail_v10["exact_matches"]
                    st.markdown("#### 1. 선택한 조성의 결과 요약")
                    m1,m2,m3 = st.columns(3)
                    m1.metric("v1 · 52개 특징 예측 (S/cm)",f"{current_v10['v1 예측 이온전도도 (S/cm)']:.4e}")
                    m2.metric("v2 · 59개 특징 예측 (S/cm)",f"{current_v10['v2 예측 이온전도도 (S/cm)']:.4e}")
                    m3.metric("두 모델 예측 비율 (큰 값/작은 값)",f"{current_v10['두 모델 예측 비율 (배)']:.2f}배")
                    u1,u2 = st.columns(2)
                    if "학습 조성 유사도 (%)" in packed["sweep"].columns:
                        u1.metric("가장 가까운 학습 조성 유사도",f"{current_v10['학습 조성 유사도 (%)']:.1f}%")
                        u2.metric("정확히 같은 조성의 학습 기록",f"{int(current_v10['동일 조성 학습 기록 (건)'])}건")
                        st.caption(f"가장 가까운 학습 소재: {current_v10['가장 가까운 학습 조성']}")
                    else:
                        st.warning("학습 참조 CSV가 없어 학습 조성 유사도와 실험 측정값을 표시할 수 없습니다.")
                    if not exact_v10.empty:
                        st.markdown("**동일 조성의 학습 측정값 · 독립 검증 자료 아님**")
                        measurements = exact_v10[["화학식","소재 계열","학습 측정값 (S/cm)"]].copy()
                        st.dataframe(measurements,use_container_width=True,hide_index=True)
                        st.caption("실험 측정값이 한 건 이상 다를 수 있습니다. 현재 모델은 구조·합성·측정 조건을 구분하지 않습니다.")
                    else:
                        st.caption("이 조성과 정확히 일치하는 학습 측정 기록은 확인되지 않았습니다.")
                    st.caption(f"학습 자료 상태: {packed['training_note']}")
                    st.divider()
                    st.markdown("#### 2. 21개 조성 예측 그래프")
                    sweep_data = packed["sweep"]
                    long_plot = sweep_data.melt(
                        id_vars=["Br 치환 비율 (%)", "화학식"],
                        value_vars=["v1 log10(S/cm)","v2 log10(S/cm)"],
                        var_name="AI 모델", value_name="예측 log10(S/cm)"
                    )
                    chart = px.line(
                        long_plot, x="Br 치환 비율 (%)", y="예측 log10(S/cm)",
                        color="AI 모델", markers=True, custom_data=["화학식"],
                        title="Cl → Br 치환 비율에 따른 두 AI 모델의 예측 변화"
                    )
                    chart.add_vline(x=chosen_pct, line_color="gray",line_dash="dash",annotation_text="선택")
                    chart.update_layout(height=420,hovermode="x unified")
                    st.plotly_chart(chart,use_container_width=True)
                    st.caption("5% 간격의 모델 예측값을 연결한 곡선입니다. 실제 실험 측정값이나 확인된 최적 조성이 아닙니다.")
                    st.divider()
                    st.markdown("#### 3. 전체 21개 조성 비교 및 보고서 다운로드")
                    cols = ["Br 치환 비율 (%)","화학식","v1 예측 이온전도도 (S/cm)","v2 예측 이온전도도 (S/cm)","두 모델 예측 비율 (배)"]
                    if "학습 조성 유사도 (%)" in sweep_data.columns:
                        cols += ["학습 조성 유사도 (%)","동일 조성 학습 기록 (건)"]
                    display = sweep_data[cols].copy()
                    for c in ["v1 예측 이온전도도 (S/cm)","v2 예측 이온전도도 (S/cm)"]:
                        display[c] = display[c].map(lambda n:f"{n:.3e}")
                    display["두 모델 예측 비율 (배)"] = display["두 모델 예측 비율 (배)"].map(lambda n:f"{n:.2f}")
                    if "학습 조성 유사도 (%)" in display.columns:
                        display["학습 조성 유사도 (%)"] = display["학습 조성 유사도 (%)"].map(lambda n:f"{n:.1f}")
                    v29_show_table(display,
                        ["Br 치환 비율 (%)", "화학식", "v1 예측 이온전도도 (S/cm)", "v2 예측 이온전도도 (S/cm)"],
                        key="v29_dashboard_table")
                    c_csv,c_pdf = st.columns(2)
                    with c_csv:
                        st.download_button(
                            "📥 21개 조성 분석 CSV",data=sweep_data.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"SSE_AI_{APP_FILE_TAG}_dashboard.csv",mime="text/csv",use_container_width=True,key="v10_csv"
                        )
                    with c_pdf:
                        try:
                            pdf_payload = v10_build_pdf(packed,detail_v10)
                        except ImportError:
                            st.error("PDF 저장에 필요한 reportlab이 없습니다. 아래 명령어로 설치 후 앱을 새로 실행해주세요.")
                            st.code("python -m pip install reportlab", language="bash")
                        except Exception as exc:
                            st.error("PDF 작성 도중 문제가 발생했습니다. 글꼴 또는 보고서 데이터를 확인해주세요.")
                            st.code(str(exc))
                        else:
                            st.download_button(
                                "📄 한국어 분석 보고서 PDF",data=pdf_payload,
                                file_name=f"SSE_AI_{APP_FILE_TAG}_Br_{chosen_pct:03d}pct.pdf",
                                mime="application/pdf",use_container_width=True,key="v10_pdf"
                            )
                    st.warning(
                        "예측 오차와 정확도는 별개입니다. v1 공식 테스트 기록(MAE 1.522, R² -0.084)은 "
                        "내부 검증보다 나빴고, v2는 공식 테스트에서 검증하지 않았습니다. "
                        "PDF나 CSV 결과를 실험 측정값으로 인용하지 마세요."
                    )
                except Exception as exc:
                    st.error("결과 화면을 표시하지 못했습니다.")
                    st.code(str(exc))
            else:
                st.info("위 '통합 분석 실행' 버튼을 누르면 분석표·곡선·PDF 저장 기능이 나타납니다.")


# ────────────────────────────────────────────────────────────────────────────
# v12 · 교수님 피드백 반영: 같은 조성의 반복 측정값을 "삭제하지 않고" 분석한다.
# 기존 모델의 학습·예측 방식은 바꾸지 않는다. 공식 테스트셋은 사용하지 않는다.
# ────────────────────────────────────────────────────────────────────────────

def v12_composition_key(formula: str) -> tuple:
    """원소 원자분율로 화학식 표기의 차이를 통합하는 비교용 키.

    reduced_formula 문자열만 비교하지 않고, 총 원자수로 정규화한 분율을
    원소 기호 순으로 정렬한다. 아주 미세한 소수 오차에 한하여 소수 9자리 사용.
    결정구조/측정 조건이 같다는 뜻은 아니다.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("빈 화학식")
    comp = Composition(formula.strip())
    total = float(comp.num_atoms)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("원자 수가 올바르지 않음")
    vector = []
    for element, amount in comp.items():
        amount = float(amount)
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("원자수 오류")
        frac = round(amount / total, 9)
        if frac > 0:
            vector.append((element.symbol, frac))
    if not vector:
        raise ValueError("원소 없음")
    return tuple(sorted(vector))


def v12_audit_records(database: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """공식 학습용 원본을 양의 정확한 측정값으로 한정해 원본 기록을 보존한다."""
    required = {FORMULA_COL, TARGET_COL}
    if not required.issubset(database.columns):
        raise ValueError("원본 CSV의 화학식 또는 이온전도도 열이 없습니다.")
    df = database.copy()
    original_rows = len(df)
    df["v12_numeric"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    numeric_positive = df["v12_numeric"].notna() & np.isfinite(df["v12_numeric"]) & df["v12_numeric"].gt(0)
    positive_rows = int(numeric_positive.sum())
    df = df.loc[numeric_positive].copy()

    keys = []
    parse_failures = 0
    for formula in df[FORMULA_COL]:
        try:
            keys.append(v12_composition_key(str(formula)))
        except (ValueError, TypeError, KeyError, OverflowError):
            keys.append(None)
            parse_failures += 1
    df["v12_key"] = keys
    df = df.loc[df["v12_key"].notna()].copy()
    df["v12_log"] = np.log10(df["v12_numeric"].astype(float))
    df["v12_doi"] = df["DOI"].fillna("").astype(str).str.strip() if "DOI" in df else ""
    meta = {
        "original_rows": original_rows,
        "positive_rows": positive_rows,
        "invalid_or_censored": original_rows - positive_rows,
        "unparsed_formulas": parse_failures,
        "usable_rows": len(df),
    }
    return df, meta


def v12_group_measurements(records: pd.DataFrame) -> pd.DataFrame:
    """원본 측정값은 유지한 채, 같은 조성의 분포를 요약한다."""
    rows = []
    for key, group in records.groupby("v12_key", sort=False):
        values = group["v12_numeric"].to_numpy(dtype=float)
        logs = np.log10(values)
        min_v, max_v = float(np.min(values)), float(np.max(values))
        median_log = float(np.median(logs))
        doi_values = {str(v).strip().lower() for v in group["v12_doi"].tolist() if str(v).strip()}
        label = str(group.iloc[0][FORMULA_COL])
        rows.append({
            "v12_key": key,
            "대표 화학식": label,
            "측정 기록 수": int(len(group)),
            "서로 다른 논문 DOI 수": len(doi_values),
            "최솟값 (S/cm)": min_v,
            "최댓값 (S/cm)": max_v,
            "로그 중앙값": median_log,
            "대표값: 로그 중앙값 역변환 (S/cm)": float(10.0 ** median_log),
            "측정값 차이 (배)": float(max_v / min_v),
            "로그 최대-최소 차이": float(logs.max() - logs.min()),
            "로그 표준편차": float(logs.std(ddof=1)) if len(group) >= 2 else np.nan,
            "원본 화학식 표기 종류": int(group[FORMULA_COL].nunique()),
        })
    return pd.DataFrame(rows).sort_values(
        ["측정 기록 수", "측정값 차이 (배)"], ascending=[False, False]
    ).reset_index(drop=True) if rows else pd.DataFrame()


def v12_check_split_overlap() -> dict:
    """기존 354/95 참조의 화학 조성 중복 여부만 확인. 분할 재생성·테스트 재평가 안 함."""
    if not TRAIN_REFERENCE_PATH.is_file() or not VALID_REFERENCE_PATH.is_file():
        return {"available": False, "note": "학습/검증 참조 CSV가 모두 있어야 분석할 수 있습니다."}
    train = pd.read_csv(TRAIN_REFERENCE_PATH, dtype={FORMULA_COL: str})
    valid = pd.read_csv(VALID_REFERENCE_PATH, dtype={FORMULA_COL: str})
    if FORMULA_COL not in train.columns or FORMULA_COL not in valid.columns:
        return {"available": False, "note": "참조 CSV에 화학식 열이 없습니다."}

    def get_keys(frame):
        keys = []
        for f in frame[FORMULA_COL].fillna(""):
            try:
                keys.append(v12_composition_key(str(f)))
            except (ValueError, TypeError, KeyError, OverflowError):
                keys.append(None)
        return keys

    tr = get_keys(train)
    va = get_keys(valid)
    tr_set = {x for x in tr if x is not None}
    va_set = {x for x in va if x is not None}
    overlap = tr_set & va_set
    examples = []
    if overlap:
        for key in sorted(overlap)[:12]:
            val_records = valid.loc[[i for i,k in enumerate(va) if k == key], FORMULA_COL]
            examples.append(str(val_records.iloc[0]))
    return {
        "available": True,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_train_distinct": len(tr_set),
        "n_valid_distinct": len(va_set),
        "n_shared_compositions": len(overlap),
        "n_valid_records_shared": sum(k in overlap for k in va if k is not None),
        "examples": examples,
    }


if selected_page == "data_quality":
    st.subheader("🧾 데이터 품질 · 동일 조성의 반복 측정값 분석")
    st.write(
        "동일 화학식의 측정값 차이에 대한 발표 피드백을 분석합니다. "
        "원래의 실험 기록과 논문 DOI를 보존하면서 통계만 계산합니다."
    )
    st.caption("원본 실험값과 논문 DOI를 조회합니다. 로그 중앙값은 기존 AI 모델에 적용되지 않은 분석용 대표값입니다.")
    st.caption("데이터: OBELiX 공식 **학습** CSV만 사용 · 공식 테스트셋은 불러오지 않음 · 단위 S/cm")

    if st.button("🔎 원본 데이터의 동일 조성·반복 측정 분석 시작", use_container_width=True, key="v12_run_audit"):
        try:
            with st.spinner("공식 학습용 측정 기록을 확인하고 화학식별로 분석하는 중..."):
                data_raw = load_database()
                data_records, data_meta = v12_audit_records(data_raw)
                data_summary = v12_group_measurements(data_records)
                if data_summary.empty:
                    raise ValueError("유효한 측정 데이터가 없습니다.")
                st.session_state["v12_audit_result"] = (data_records, data_meta, data_summary)
            st.success("데이터 품질 분석을 완료했습니다.")
        except Exception as exc:
            st.session_state.pop("v12_audit_result", None)
            st.error("분석에 실패했습니다. 인터넷 연결 또는 OBELiX_train.csv 파일을 확인해 주세요.")
            st.code(str(exc))

    if "v12_audit_result" in st.session_state:
        audit_records, audit_meta, audit_summary = st.session_state["v12_audit_result"]
        repeated = audit_summary[audit_summary["측정 기록 수"] >= 2].copy()
        repeated_records = int(repeated["측정 기록 수"].sum()) if len(repeated) else 0
        extra_records = repeated_records - len(repeated)

        st.markdown("#### 1. 원본 데이터 · 반복 측정 현황")
        a, b, c, d = st.columns(4)
        a.metric("원본 학습 기록", f"{audit_meta['original_rows']}건")
        b.metric("양의 수치형 기록", f"{audit_meta['positive_rows']}건")
        c.metric("서로 다른 원소 조성", f"{len(audit_summary)}종")
        d.metric("반복 측정 조성", f"{len(repeated)}종")
        st.caption(
            f"분석 가능한 기록 {len(audit_records)}건 · 반복 측정 조성에 포함된 기록 {repeated_records}건 · "
            f"첫 기록을 제외한 추가 측정 기록 {extra_records}건 · "
            f"검출 한계/범위/0/비수치 등 제외 {audit_meta['invalid_or_censored']}건 · "
            f"화학식 해석 제외 {audit_meta['unparsed_formulas']}건"
        )
        st.caption("같은 원소 조성의 복수 기록은 서로 다른 실험·논문일 수 있으며 자동으로 제거하지 않았습니다.")


        st.markdown("#### 2. 화학식 검색 · 실제 문헌 측정값")
        st.caption(
            "유효 측정값이 있는 362종 전체에서 원소 비율이 같은 조성을 찾습니다. "
            "화학식 순서와 괄호 표기가 달라도 같은 원소 비율이면 같은 조성으로 조회합니다. "
            "반복 측정 여부와 관계없이 검색할 수 있습니다."
        )
        formula_search = st.text_input(
            "분석할 화학식 (정확한 조성 검색)",
            value="Li6PS5Cl",
            placeholder="예: Li6PS5Cl 또는 Li10Ge(PS6)2",
            key="v12_formula_search",
        )
        query_key = None
        if formula_search.strip():
            try:
                query_key = v12_composition_key(normalize_user_formula(formula_search))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                st.error(f"화학식 해석 실패: {exc}")
        else:
            st.info("화학식을 입력하면 해당 조성의 실제 학습 측정 기록을 조회합니다.")

        matching = (
            audit_summary[audit_summary["v12_key"].map(lambda key: key == query_key)]
            if query_key is not None else audit_summary.iloc[0:0]
        )
        if query_key is not None and matching.empty:
            st.warning(
                "해당 원소 조성의 유효한 양의 수치형 측정 기록이 OBELiX 학습 자료에서 발견되지 않았습니다. "
                "이전 검색 결과를 대신 표시하지 않습니다. "
                "원본에 값이 없거나 검출한계·비수치 기록만 존재할 수도 있습니다."
            )
        elif not matching.empty:
            selected = matching.iloc[0]
            same_comp = audit_records[audit_records["v12_key"] == selected["v12_key"]].copy()
            count = len(same_comp)
            ratio = float(selected["측정값 차이 (배)"])
            st.success(f"검색된 조성: {selected['대표 화학식']} · 유효 문헌 측정 기록 {count}건")
            m1, m2, m3 = st.columns(3)
            m1.metric("문헌 측정 기록", f"{count}건")
            m2.metric("최댓값/최솟값", f"{ratio:.2f}배" if count >= 2 else "비교 불가 (1건)")
            m3.metric(
                "로그 중앙값 대표 전도도",
                f"{selected['대표값: 로그 중앙값 역변환 (S/cm)']:.3e} S/cm",
            )
            if count == 1:
                st.caption("측정값이 1건이므로 서로 다른 실험값의 차이를 계산할 수 없습니다. 중앙값은 해당 1건과 같습니다.")
            else:
                st.caption("같은 원소 조성의 문헌 측정값은 서로 달라도 모두 원본 그대로 표시합니다.")

            chart_data = same_comp[["v12_log", "v12_numeric", "v12_doi"]].copy().reset_index(drop=True)
            chart_data["측정 번호"] = [f"측정 {i + 1}" for i in range(len(chart_data))]
            fig = px.scatter(
                chart_data, x="측정 번호", y="v12_log",
                hover_data={"v12_numeric": ":.4e", "v12_doi": True},
                labels={
                    "v12_log": "문헌 측정 이온전도도 log₁₀(S/cm)",
                    "v12_numeric": "실험값 (S/cm)",
                    "v12_doi": "논문 DOI",
                },
                title="해당 조성의 원본 실험 측정값",
            )
            fig.update_traces(marker=dict(size=14, color="#26a69a"))
            fig.add_hline(
                y=float(selected["로그 중앙값"]),
                line_dash="dash", line_color="#ffb74d", annotation_text="로그 중앙값",
            )
            fig.update_layout(height=350, margin=dict(l=10, r=15, t=55, b=25))
            st.plotly_chart(fig, use_container_width=True)

            detail_columns = [
                c for c in ["ID", FORMULA_COL, TARGET_COL, "Family", "Space group", "DOI", "note"]
                if c in same_comp.columns
            ]
            detailed = same_comp[detail_columns].copy()
            if "DOI" in detailed.columns:
                detailed["논문 링크"] = detailed["DOI"].fillna("").astype(str).map(
                    lambda doi: f"https://doi.org/{doi.strip()}" if doi.strip().startswith("10.") else ""
                )
            config = (
                {"논문 링크": st.column_config.LinkColumn("논문 원문", display_text="원문 보기")}
                if "논문 링크" in detailed.columns else {}
            )
            st.dataframe(detailed, use_container_width=True, hide_index=True, column_config=config)
            st.caption(
                "문헌 실험 측정값과 논문 출처를 보여주는 분석 화면입니다. "
                "AI 예측값이나 동일한 결정구조라는 증거는 아닙니다."
            )


        with st.expander("반복 측정 조성 순위표 · 연구용 상세", expanded=False):
            st.markdown("#### 2. 측정값이 서로 다른 조성 찾기")
            if repeated.empty:
                st.info("현재 정확한 양의 측정값 자료에서 반복 측정 조성을 찾지 못했습니다.")
            else:
                left, right = st.columns([2,1])
                with left:
                    sort_choice = st.selectbox(
                        "정렬 방식", ["측정값 차이 큰 순", "측정 기록 많은 순"], key="v12_sort"
                    )
                with right:
                    only_diff = st.checkbox("서로 다른 측정값만", value=True, key="v12_diff_only")
                filtered_groups = repeated.copy()
                if only_diff:
                    filtered_groups = filtered_groups[filtered_groups["측정값 차이 (배)"] > (1.0 + 1e-10)]
                if sort_choice == "측정값 차이 큰 순":
                    filtered_groups = filtered_groups.sort_values("측정값 차이 (배)", ascending=False)
                else:
                    filtered_groups = filtered_groups.sort_values(
                        ["측정 기록 수", "측정값 차이 (배)"], ascending=False
                    )

                show_groups = filtered_groups[[
                    "대표 화학식", "측정 기록 수", "서로 다른 논문 DOI 수",
                    "최솟값 (S/cm)", "최댓값 (S/cm)",
                    "측정값 차이 (배)", "대표값: 로그 중앙값 역변환 (S/cm)",
                ]].head(60).copy()
                for name in ["최솟값 (S/cm)", "최댓값 (S/cm)", "대표값: 로그 중앙값 역변환 (S/cm)"]:
                    show_groups[name] = show_groups[name].map(lambda n: f"{n:.3e}")
                show_groups["측정값 차이 (배)"] = show_groups["측정값 차이 (배)"].map(lambda n: f"{n:.2f}")
                v29_show_table(show_groups,
                    ["대표 화학식", "측정 기록 수", "측정값 차이 (배)", "대표값: 로그 중앙값 역변환 (S/cm)"],
                    key="v29_repeat_table")
                st.caption("상위 60종만 화면에 표시합니다. 다운로드 CSV에는 전체 조성 분석이 포함됩니다.")


        with st.expander("모델 학습 방식·중복 검토·원본 CSV 저장", expanded=False):
            st.divider()
            st.markdown("#### 4. 현재 모델은 이 차이를 어떻게 처리하나?")
            st.write(
                "현재 v1은 원소 조성비 **52개**, v2는 원소 조성비와 물성 특징 **59개**를 입력받습니다. "
                "같은 조성은 모델에 같은 특징값으로 들어가기 때문에, 서로 다른 측정 조건의 차이를 "
                "구분해 별도 전도도를 출력할 수 없습니다."
            )
            st.write(
                "현재 모델의 학습 단계에서는 기록마다 하나의 목표 전도도를 사용했습니다. "
                "동일 조성에 서로 다른 목표값이 존재하면 모델은 패턴을 학습하며 타협적인 값을 예측할 수 있습니다. "
                "**이번 v12의 중앙값은 기존 AI 모델의 내부 계산식이나 기존 학습값을 변경하지 않습니다.**"
            )
            st.markdown("#### 5. 학습/검증 조성 겹침 확인 (참고)")
            try:
                split_info = v12_check_split_overlap()
                if not split_info["available"]:
                    st.info(split_info["note"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("학습 참조 기록", f"{split_info['n_train']}건")
                    c2.metric("내부 검증 참조 기록", f"{split_info['n_valid']}건")
                    c3.metric("양쪽에 나타난 동일 조성", f"{split_info['n_shared_compositions']}종")
                    st.caption(
                        f"내부 검증에서 학습 조성과 겹치는 기록 {split_info['n_valid_records_shared']}건. "
                        "참조 CSV의 화학식만 비교했으며 논문별 중복은 검증하지 못했습니다."
                    )
                    if split_info["n_shared_compositions"]:
                        st.warning(
                            "학습과 내부 검증 양쪽에 같은 조성이 있습니다. 조성비만 입력하는 모델은 "
                            "내부 검증 성능이 낙관적으로 보일 수 있어, 향후 조성·논문을 묶어 분리하는 검증이 필요합니다."
                        )
                        st.write("겹치는 조성 예시: " + ", ".join(split_info["examples"]))
                    else:
                        st.info(
                            "두 참조 CSV에서는 동일 조성 겹침이 발견되지 않았습니다. "
                            "다만 같은 논문의 유사 조성 여부까지 증명한 결과는 아닙니다."
                        )
            except Exception as exc:
                st.warning("학습/검증 분할 겹침 분석을 수행하지 못했습니다.")
                st.code(str(exc))

            st.divider()
            st.markdown("#### 6. 전체 결과 CSV 및 대표값 후보 다운로드")
            summary_csv = audit_summary.drop(columns=["v12_key"]).copy()
            representative = summary_csv[[
                "대표 화학식", "측정 기록 수", "서로 다른 논문 DOI 수",
                "대표값: 로그 중앙값 역변환 (S/cm)", "최솟값 (S/cm)",
                "최댓값 (S/cm)", "측정값 차이 (배)",
            ]].copy()
            representative.insert(1, "대표 전도도의 log10", np.log10(representative["대표값: 로그 중앙값 역변환 (S/cm)"]))
            original_csv = audit_records.drop(columns=["v12_key", "v12_doi"]).copy()
            x1, x2, x3 = st.columns(3)
            x1.download_button(
                "📥 원본 유효 측정 기록", data=original_csv.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_v12_original_valid_measurements.csv", mime="text/csv", use_container_width=True
            )
            x2.download_button(
                "📥 조성별 분포 통계", data=summary_csv.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_v12_duplicate_composition_summary.csv", mime="text/csv", use_container_width=True
            )
            x3.download_button(
                "📥 대표값 학습 후보 데이터", data=representative.to_csv(index=False).encode("utf-8-sig"),
                file_name="SSE_v12_logmedian_training_candidates.csv", mime="text/csv", use_container_width=True
            )
            st.warning(
                "대표값 후보 데이터는 아직 모델 학습이나 검증에 사용하면 안 됩니다. "
                "먼저 학습/검증 그룹을 나누고, 논문별 측정조건·공간군의 차이를 확인한 뒤 "
                "훈련 구간에만 대표값 집계를 적용해야 데이터 누출을 방지할 수 있습니다."
            )
    else:
        st.info("위 분석 버튼을 누르면 실제 원본 학습 데이터에서 반복 측정 현황을 계산합니다.")

# ─────────────────────────────────────────────────────────────────────────────
# v15: 과거 Colab에서 수행한 교차검증 결과의 읽기 전용 리포트
# 이 페이지에서는 모델을 학습하거나 새로운 테스트 데이터를 불러오지 않습니다.
# ─────────────────────────────────────────────────────────────────────────────
V15_CV_FILES = {
    "summary": BASE_DIR / "SSE_v14_cv_summary.csv",
    "fold": BASE_DIR / "SSE_v14_fold_scores.csv",
    "family": BASE_DIR / "SSE_v14_family_errors.csv",
    "oof": BASE_DIR / "SSE_v14_oof_predictions.csv",
}


@st.cache_data(show_spinner=False)
def v15_read_cv_tables():
    """사용자가 Colab에서 생성한 기존 v14 CSV만 읽는 읽기 전용 함수."""
    required = {
        "summary": {"검증방식", "모델코드", "폴드수", "검증총기록", "전체OOF_MAE_log10", "전체OOF_R2", "폴드_MAE_평균", "폴드_MAE_표준편차"},
        "fold": {"검증방식", "폴드", "모델코드", "검증기록", "DOI겹침", "MAE_log10", "R2"},
        "family": {"검증방식", "모델코드", "소재계열", "기록수", "MAE_log10"},
        "oof": {"검증방식", "모델코드", "폴드", "화학식", "실제_log10", "예측_log10", "절대오차_log10"},
    }
    result = {}
    for name, path in V15_CV_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"결과 CSV 파일이 없습니다: {path.name}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing_cols = required[name] - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{path.name}에 필요한 열이 없습니다: {', '.join(sorted(missing_cols))}")
        result[name] = frame
    expected_models = {"A", "B", "C", "BASE"}
    if set(result["summary"]["모델코드"]) != expected_models:
        raise ValueError("교차검증 모델 구성(A, B, C, BASE)을 확인해주세요.")
    if result["summary"]["검증방식"].nunique() != 2:
        raise ValueError("두 검증방식의 결과가 모두 들어 있어야 합니다.")
    return result


if selected_page == "validation" and validation_view == "cv":
    st.subheader("📈 AI 모델 개선 실험 · 5겹 교차검증")
    st.write("중복 조성과 논문 출처를 함께 분리하여, 52개 특징(A)·59개 특징(B)·반복값 중앙값(C)의 학습 방식을 비교한 과거 5겹 교차검증 결과입니다.")
    st.caption("개발 자료 449건의 별도 교차검증 실험이며, 현재 저장된 v1·v2 모델을 다시 학습하거나 공식 테스트를 재평가하지 않습니다.")
    with st.expander("검증을 개선한 이유와 A·B·C 모델 설명", expanded=False):
        st.write("기존 95건 내부 검증에는 학습과 조성이 겹친 기록이 있었습니다. 이번 실험은 같은 조성이나 논문으로 연결된 기록을 함께 분리하여 점검합니다.")
        st.write("**A:** 원소 조성비 52개 · **B:** 원소 물성 7개 추가한 59개 특징 · **C:** 학습 구간에서 반복 측정값을 로그 중앙값으로 정리한 52개 특징 모델")
        st.write("**BASE:** 학습 구간 로그 전도도 중앙값만 예측하는 기준선입니다. **MAE**는 낮을수록, **R²**는 대체로 높을수록 좋습니다.")
        st.write("**5겹 교차검증:** 데이터를 다섯 묶음으로 나누어 한 묶음씩 번갈아 평가합니다.")

    missing_cv = [p.name for p in V15_CV_FILES.values() if not p.is_file()]
    if missing_cv:
        st.warning("교차검증 결과 CSV가 아직 없습니다. 업데이트 ZIP 안의 다음 파일을 app.py와 같은 폴더에 넣어주세요: " + ", ".join(missing_cv))
    else:
        try:
            cv_data = v15_read_cv_tables()
        except Exception as exc:
            st.error("교차검증 기록을 불러오지 못했습니다. CSV 파일 구성을 확인해 주세요.")
            st.code(str(exc))
        else:
            summary = cv_data["summary"]
            folds = cv_data["fold"]
            families = cv_data["family"]
            predictions = cv_data["oof"]
            name_by_code = {"A": "A · 52개 특징", "B": "B · 59개 특징", "C": "C · 로그 중앙값", "BASE": "기준선"}
            st.markdown("#### 1. 어떤 데이터 분리 방식으로 검증했나요?")
            method_labels = {
                "조성+DOI 연결 분리": "조성 + 논문(DOI) 연결 그룹 분리 · 더 엄격한 조건",
                "조성 분리": "동일 조성 분리 · 논문 출처는 겹칠 수 있는 조건",
            }
            selected_method = st.radio(
                "검증 방법 선택", list(method_labels), format_func=lambda k: method_labels[k],
                horizontal=False, key="v15_cv_method",
            )
            st.caption(
                "조성 분리는 같은 원소 조성이 학습·검증에 함께 나타나지 않도록 합니다. "
                "조성+DOI 연결 분리는 같은 조성 또는 논문으로 연결된 기록을 하나의 그룹으로 묶어 분리합니다. "
                "그룹 크기가 불균등할 수 있습니다."
            )
            chosen = summary[summary["검증방식"] == selected_method].copy()
            chosen["설명"] = chosen["모델코드"].map(name_by_code)
            chosen = chosen.set_index("모델코드").reindex(["A", "B", "C", "BASE"]).reset_index()
            if chosen["전체OOF_MAE_log10"].isna().any():
                st.error("선택한 방법에 일부 모델 결과가 빠져 있습니다.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("검증한 원본 측정 기록", f"{int(chosen['검증총기록'].max()):,}건")
                c2.metric("교차검증", "5겹")
                c3.metric("가장 낮은 전체 MAE", f"{chosen.loc[chosen['모델코드'].isin(['A','B','C']), '전체OOF_MAE_log10'].min():.3f}")
                shown_cv = chosen[["설명", "전체OOF_MAE_log10", "전체OOF_R2", "폴드_MAE_평균", "폴드_MAE_표준편차"]].copy()
                shown_cv.columns = ["학습 방식", "전체 MAE (log10) ↓", "전체 R² ↑", "5개 폴드 MAE 평균 ↓", "폴드별 MAE 표준편차 ↓"]
                st.dataframe(shown_cv.round(3), hide_index=True, use_container_width=True)
                st.caption("전체 MAE/R²는 5개 검증 묶음의 모든 예측을 합쳐 계산한 값입니다. 폴드 평균과는 다를 수 있습니다.")

                view = chosen[chosen["모델코드"].isin(["A", "B", "C"])].copy()
                fig_mae = px.bar(
                    view, x="설명", y="전체OOF_MAE_log10", text="전체OOF_MAE_log10",
                    title="학습 방식별 전체 교차검증 MAE (낮을수록 좋음)",
                    labels={"설명": "학습 방식", "전체OOF_MAE_log10": "MAE (log10)"},
                )
                fig_mae.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
                fig_mae.update_yaxes(range=[0, float(view["전체OOF_MAE_log10"].max()) * 1.23], zeroline=True)
                fig_mae.update_layout(height=370, margin=dict(l=20, r=15, t=60, b=25))
                st.plotly_chart(fig_mae, use_container_width=True)
                if selected_method == "조성+DOI 연결 분리":
                    st.info(
                        "이 조건에서는 A의 전체 MAE가 약 1.289, C가 약 1.297, B가 약 1.397입니다. "
                        "A와 C의 차이는 작으며, B가 더 좋다는 결과는 재현되지 않았습니다. "
                        "이는 **정확도 개선을 입증한 결과가 아니라 데이터 처리 방식의 효과를 점검한 결과**입니다."
                    )
                else:
                    st.info(
                        "동일 조성을 분리한 조건에서 A·B·C의 전체 MAE는 모두 약 0.83입니다. "
                        "한 방법이 뚜렷하게 우수하다고 보기 어렵습니다."
                    )
                with st.expander("검증 묶음별 오차 변화 · 자세히 보기", expanded=False):
                    st.markdown("#### 2. 검증 묶음에 따라 오차가 얼마나 변하나요?")
                    fold_view = folds[(folds["검증방식"] == selected_method) & (folds["모델코드"].isin(["A", "B", "C"]))].copy()
                    fold_view["학습 방식"] = fold_view["모델코드"].map(name_by_code)
                    fig_fold = px.line(
                        fold_view, x="폴드", y="MAE_log10", color="학습 방식", markers=True,
                        title="5개 검증 묶음별 MAE",
                        labels={"폴드": "검증 묶음 번호", "MAE_log10": "MAE (log10)"},
                    )
                    fig_fold.update_xaxes(dtick=1)
                    fig_fold.update_layout(height=380, margin=dict(l=20, r=15, t=58, b=35))
                    st.plotly_chart(fig_fold, use_container_width=True)
                    sizes = folds[(folds["검증방식"] == selected_method) & (folds["모델코드"] == "A")]
                    if not sizes.empty:
                        st.caption("검증 묶음별 기록 수: " + ", ".join(str(int(v)) for v in sizes.sort_values("폴드")["검증기록"]) + "건")
                        if selected_method == "조성+DOI 연결 분리":
                            doi_overlap = sizes["DOI겹침"]
                            st.caption(f"조성·DOI 연결 분리에서 폴드별 보고된 DOI 중복: {', '.join(str(int(v)) for v in doi_overlap)}건")
                    st.warning(
                        "이번 실험은 **기존 OBELiX 학습 449건 안에서의 교차검증**입니다. "
                        "이미 한 차례 확인한 공식 테스트셋을 이번 모델 선정에 사용하지 않았습니다. "
                        "새로운 독립 테스트에서의 예측 성능이나 실제 배터리 성능을 보증하지 않습니다."
                    )

                with st.expander("소재 계열별 오차 및 큰 오류 사례 · 연구용 상세", expanded=False):
                    st.markdown("#### 3. 소재 계열별 오차 살펴보기")
                    st.caption("기록 수가 적은 계열의 MAE는 크게 흔들릴 수 있어 기본적으로 5건 이상인 계열만 표시합니다.")
                    family_model = st.selectbox("분석할 학습 방식", ["A", "B", "C"], format_func=lambda c: name_by_code[c], key="v15_family_model")
                    min_family_n = st.selectbox("최소 평가 기록 수", [5, 10, 20], key="v15_min_family_n")
                    family_view = families[
                        (families["검증방식"] == selected_method)
                        & (families["모델코드"] == family_model)
                        & (families["기록수"] >= min_family_n)
                    ].sort_values("MAE_log10", ascending=False).head(15)
                    if family_view.empty:
                        st.info("이 조건에 해당하는 소재 계열이 없습니다.")
                    else:
                        shown_f = family_view[["소재계열", "기록수", "MAE_log10"]].copy()
                        shown_f.columns = ["소재 계열", "평가 기록 수", "MAE (log10)"]
                        shown_f["소재 계열"] = shown_f["소재 계열"].fillna("미분류").replace({"None": "미분류", "nan": "미분류", "": "미분류"})
                        st.dataframe(shown_f.round(3), hide_index=True, use_container_width=True)

                    st.markdown("#### 4. 개별 오차가 큰 사례 살펴보기")
                    example_view = predictions[
                        (predictions["검증방식"] == selected_method)
                        & (predictions["모델코드"] == family_model)
                    ].sort_values("절대오차_log10", ascending=False).head(10)
                    if not example_view.empty:
                        cols = [c for c in ["화학식", "소재계열", "폴드", "실제_log10", "예측_log10", "절대오차_log10"] if c in example_view.columns]
                        st.dataframe(example_view[cols].round(3), hide_index=True, use_container_width=True)
                        st.caption("오차가 큰 사례를 보여주는 진단용 표이며, 이 결과로 모델을 반복 조정하면 검증 점수가 낙관적으로 변할 수 있습니다.")

                with st.expander("검증 원본 CSV 다운로드", expanded=False):
                    st.markdown("#### 5. 검증 원본 결과 다운로드")
                    for name, title in [("summary", "전체 성능 요약"), ("fold", "폴드별 성능"), ("family", "소재 계열별 오차"), ("oof", "검증 예측 상세")]:
                        st.download_button(
                            f"📥 {title} CSV",
                            data=cv_data[name].to_csv(index=False).encode("utf-8-sig"),
                            file_name=V15_CV_FILES[name].name, mime="text/csv",
                            key=f"v15_download_{name}",
                        )
                    st.caption("데이터 출처: Colab 5겹 교차검증 출력 CSV · 파일 생성 시의 모델 훈련 설정 및 데이터 분할을 보존한 기록입니다.")



# ─────────────────────────────────────────────────────────────────────────────
# v27: 결정구조 메타데이터 조회 및 구조 특징 기반 교차검증 결과 (모델 예측 교체 아님)
# ─────────────────────────────────────────────────────────────────────────────
V27_CATALOG_PATH = BASE_DIR / "SSE_v27_structure_catalog.csv"
V27_CV_SUMMARY = BASE_DIR / "SSE_v27_structure_cv_summary.csv"
V27_FOLD_SCORES = BASE_DIR / "SSE_v27_structure_fold_scores.csv"

@st.cache_data(show_spinner=False)
def v27_load_structure_data():
    required_files = (V27_CATALOG_PATH, V27_CV_SUMMARY, V27_FOLD_SCORES)
    missing = [p.name for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError("필수 구조 검토 파일이 없습니다: " + ", ".join(missing))
    catalog = pd.read_csv(V27_CATALOG_PATH, dtype={"formula": str, "composition_key": str, "ID": str}).fillna({"doi": "", "cif_status": "not recorded", "structure_notes": "", "close_match": "", "cif_reference": "", "family": "", "doi_link": ""})
    summary = pd.read_csv(V27_CV_SUMMARY)
    folds = pd.read_csv(V27_FOLD_SCORES)
    expected = {"source_row", "formula", "composition_key", "spacegroup_number", "cif_status", "doi", "a", "b", "c", "alpha", "beta", "gamma", "Z"}
    if not expected.issubset(catalog.columns):
        raise ValueError(f"구조 카탈로그의 필수 열이 누락됐습니다: {sorted(expected - set(catalog.columns))}")
    if len(catalog) != 449 or catalog['source_row'].duplicated().any():
        raise ValueError('구조 카탈로그가 기존 449개 개발 기록과 일치하지 않습니다.')
    return catalog, summary, folds


def v27_composition_key(formula):
    composition = Composition(normalize_user_formula(formula))
    if composition.num_atoms <= 0:
        raise ValueError('원소의 원자 수가 양수여야 합니다.')
    # v26 조성키는 원자 비율을 소수점 9자리로 인코딩한 값입니다.
    fractions = composition.fractional_composition.get_el_amt_dict()
    return "|".join(f"{element}:{float(frac):.9f}" for element, frac in sorted(fractions.items()))


if selected_page == "structure":
    st.subheader('🧬 결정구조 정보와 구조 기반 예측 가능성')
    st.write('동일 화학식에서도 공간군과 격자 정보가 다를 수 있습니다. 문헌 구조 기록을 비교하고 연구용 모델의 교차검증 결과를 확인합니다.')
    st.caption("현재 화학식 예측 모델은 결정구조를 입력받지 않습니다. 구조 특징 모델은 별도 사전 검증 결과로만 보여줍니다.")
    try:
        v27_catalog, v27_cv, v27_folds = v27_load_structure_data()
    except Exception as v27_exc:
        st.error(f'구조 데이터 파일을 읽지 못했습니다: {v27_exc}')
        st.caption('업데이트 ZIP에서 app.py와 CSV 3개를 모두 같은 폴더에 넣어주세요.')
    else:
        st.markdown('### 1. 어떤 구조 정보가 수집되어 있나요?')
        sg_by_formula = v27_catalog.groupby('composition_key')['spacegroup_number'].nunique()
        cif_status = v27_catalog['cif_status'].value_counts()
        col1,col2,col3,col4 = st.columns(4)
        col1.metric('실험 전도도 기록', f'{len(v27_catalog)}건')
        col2.metric('서로 다른 조성', f'{v27_catalog.composition_key.nunique()}종')
        col3.metric('복수 공간군 기록 조성', f'{int((sg_by_formula>1).sum())}종')
        col4.metric('CIF 자료 이용 불가', f"{int(cif_status.get('not available',0))}건")
        st.caption(f"CIF 처리 완료 {int(cif_status.get('done',0))}건 · 자료 이용 불가 {int(cif_status.get('not available',0))}건 · 상태 미기록 {int(cif_status.get('not recorded',0))}건. 처리 완료는 측정 시료와 구조의 일치를 보증하지 않습니다.")
        st.caption('구조 숫자가 기록되어도 실제 전도도 측정 시료와 동일한 구조인지 원문 확인이 필요합니다.')

        st.markdown('### 2. 관심 화학식의 구조 기록 살펴보기')
        st.caption('예시: Li7La3Zr2O12 (동일 조성에서 다른 공간군 기록) · Li3PS4 · Li6PS5Cl. 일반 숫자·아래첨자·대소문자 입력을 지원합니다.')
        inp = st.text_input('구조 기록을 찾아볼 화학식', value='Li7La3Zr2O12', key='v27_structure_formula')
        try:
            ckey = v27_composition_key(inp)
            matches = v27_catalog[v27_catalog['composition_key'] == ckey].copy()
        except Exception as exc:
            st.error('화학식 해석 실패: ' + str(exc))
            matches = pd.DataFrame()
        if matches.empty:
            st.info('현재 보관된 449개 개발 기록에는 이 원소 조성과 일치하는 구조가 없습니다. 구조값을 임의로 채우지 않습니다.')
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric('같은 조성의 문헌 기록', f'{len(matches)}건')
            m2.metric('서로 다른 공간군', f"{matches['spacegroup_number'].nunique()}종")
            m3.metric('서로 다른 격자상수 a', f"{matches['a'].nunique()}종")
            if matches['spacegroup_number'].nunique() > 1:
                st.warning('같은 원소 조성에 다른 공간군이 기록돼 있어요. **화학식만으로 어떤 구조를 선택할지 알 수 없습니다.**')
            with st.expander('문헌별 구조 기록·공간군·격자상수 상세 보기', expanded=False):
                viewcols=['ID','formula','spacegroup','spacegroup_number','a','b','c','alpha','beta','gamma','Z','cif_status','doi']
                shown_structure = matches[viewcols].copy()
                shown_structure['cif_status'] = shown_structure['cif_status'].map({
                    'done': '처리 완료', 'not available': '자료 이용 불가', 'not recorded': '상태 미기록',
                }).fillna(shown_structure['cif_status'])
                st.dataframe(shown_structure, hide_index=True, use_container_width=True)
                labels={int(r.source_row):f"ID {r.ID} · SG {int(r.spacegroup_number)} · {r.formula} · DOI {r.doi}" for r in matches.itertuples()}
                chosen_row=st.selectbox('원본 기록 상세 보기',list(labels),format_func=lambda i:labels[i],key='v27_struct_record')
                row=v27_catalog.loc[v27_catalog['source_row']==chosen_row].iloc[0]
                with st.container(border=True):
                    st.markdown(f"**원본 ID:** `{row['ID']}` · **원소 조성:** `{row['formula']}`")
                    st.write(f"**공간군:** {row.get('spacegroup','')} (No. {int(row['spacegroup_number'])})")
                    st.write(f"**격자 a,b,c:** {row['a']:.4f}, {row['b']:.4f}, {row['c']:.4f} Å · **격자각:** {row['alpha']:.3f}°, {row['beta']:.3f}°, {row['gamma']:.3f}° · **Z:** {int(row['Z'])}")
                    st.write(f"**CIF 처리 상태:** `{ {'done': '처리 완료', 'not available': '자료 이용 불가', 'not recorded': '상태 미기록'}.get(row['cif_status'], row['cif_status']) }` · **CIF 출처 메모:** `{str(row.get('cif_reference','')) or '(기재 없음)'}`")
                    if row.get('structure_notes'):
                        st.write(f"**원본 주석:** {row['structure_notes']}")
                    if str(row.get('close_match','')).lower() == 'yes':
                        st.warning('원본에 close match가 표시된 기록입니다. 구조의 정확한 조성 대응 관계를 원문에서 확인해야 합니다.')
                    if row.get('doi_link'):
                        st.markdown(f"[논문 DOI 출처 열기]({row['doi_link']})")
                    st.caption('DOI 출처와 격자 정보는 참고 기록입니다. 특정 격자 정보가 그 전도도 측정 시료 자체를 나타낸다고 자동으로 확정하지 않습니다.')

        with st.expander('구조 특징 모델 4종 · 5겹 교차검증 결과 (연구용)', expanded=False):
            st.markdown('### 3. 구조 특징을 추가한 모델의 사전 검증')
            st.caption('기존 개발 데이터 449건에 대해 별도로 재학습한 5겹 교차검증 결과이며, 현재 앱에 로드된 v1·v2 모델의 직접 비교 성능이 아닙니다.')
            vmethod=st.radio('검증 자료 분리 방식', ['조성+DOI 연결 분리', '조성 분리'],format_func=lambda x: '조성 + 논문 출처 함께 분리' if 'DOI' in x else '조성만 분리',key='v27_cv_method')
            a=v27_cv[v27_cv['검증방식']==vmethod].copy()
            renames={'A52':'조성만 사용 (52개)','LATTICE':'격자 추가 (61개)','SPACEGROUP':'공간군 추가 (97개)','BOTH':'격자+공간군 (106개)'}
            a['학습 방식']=a['모델'].map(renames)
            if len(a)==4:
                show=a[['학습 방식','특징개수','전체OOF_MAE','전체OOF_R2','폴드평균_MAE','폴드표준편차_MAE']].copy()
                show.columns=['모델','특징 수','MAE (낮을수록 좋음)','R² (높을수록 좋음)','5개 폴드 MAE 평균','폴드 MAE 표준편차']
                st.dataframe(show.round(3),hide_index=True,use_container_width=True)
                fig=px.bar(a,x='학습 방식',y='전체OOF_MAE',text='전체OOF_MAE',title='모델별 5겹 교차검증 MAE',labels={'학습 방식':'학습 방식','전체OOF_MAE':'MAE(log10)'})
                fig.update_traces(texttemplate='%{text:.3f}',textposition='outside',cliponaxis=False)
                fig.update_yaxes(range=[0,float(a['전체OOF_MAE'].max())*1.2])
                fig.update_layout(height=380,margin=dict(l=20,r=10,t=60,b=35))
                st.plotly_chart(fig,use_container_width=True)
                vfold=v27_folds[v27_folds['검증방식']==vmethod].copy()
                vfold['학습 방식']=vfold['모델'].map(renames)
                fig2=px.line(vfold,x='폴드',y='MAE_log10',color='학습 방식',markers=True,title='5개 검증 묶음별 오차 변화',labels={'폴드':'검증 묶음 번호','MAE_log10':'MAE(log10)'})
                fig2.update_xaxes(dtick=1)
                fig2.update_layout(height=370,margin=dict(l=20,r=10,t=60,b=30))
                st.plotly_chart(fig2,use_container_width=True)
                st.info('조성·논문 동시 분리에서 격자만 추가한 모델은 **MAE 약 1.233 / R² 약 0.466**, 공간군만 추가한 모델은 **MAE 약 1.202 / R² 약 0.404**였어요. 서로 다른 지표에서 장점이 다르며, 여러 방식 탐색 후 얻은 결과이므로 독립 검증에서의 우수성을 보증하지 않습니다.' if 'DOI' in vmethod else '조성만 분리할 때의 검증 점수는 논문 출처까지 분리했을 때보다 낙관적으로 나올 수 있습니다.')
                st.download_button('📥 구조 특징 교차검증 결과 CSV',data=v27_cv.to_csv(index=False).encode('utf-8-sig'),file_name='SSE_v27_structure_cv_summary.csv',mime='text/csv',key='v27_cv_download')
            else:
                st.error('교차검증 결과의 모델 4개가 모두 존재하지 않습니다. CSV 파일을 확인해 주세요.')

        with st.expander('구조 기반 AI 예측을 실제로 적용하기 위한 조건', expanded=False):
            st.markdown('### 4. 연구용 구조 모델을 실제로 사용하기 위한 조건')
            st.write('① 예측하려는 **실제 시료의** 구조 또는 출처가 확인되어야 합니다. ② 구조 정보가 유효한 훈련 데이터로 별도 모델을 학습해야 합니다. ③ 기존 조성 모델과 공정한 평가를 마친 뒤에만 기능을 연결해야 합니다.')
            st.warning('이 **구조 정보 조회 메뉴**에서는 구조 모델로 예측하지 않습니다. 별도 **구조 기반 시범 예측** 메뉴에서 모델 파일을 준비한 뒤 연구용으로 시험할 수 있으며, 화학식만 가진 경우에는 기존 **AI 예측 · 소재 비교** 메뉴를 이용해 주세요.')



    # ─────────────────────────────────────────────────────────────────────────────
# v28: separate research-only lattice model (only after user trains it with sklearn 1.6.1)
# ─────────────────────────────────────────────────────────────────────────────
V28_MODEL_PATH = BASE_DIR / 'SSE_AI_Structure_Model_v28.joblib'
V28_CV_PATH = BASE_DIR / 'SSE_v28_screened_cv_summary.csv'


@st.cache_resource(show_spinner=False)
def v28_load_pilot_model():
    if not V28_MODEL_PATH.is_file():
        raise FileNotFoundError('SSE_AI_Structure_Model_v28.joblib 파일이 없습니다. Colab에서 학습 후 같은 폴더에 복사해주세요.')
    # Trust only files created by your own Colab training notebook.
    saved = joblib.load(V28_MODEL_PATH)
    if not isinstance(saved, dict) or saved.get('feature_type') != 'composition_lattice_9':
        raise ValueError('v28 구조 모델 파일 형식이 다릅니다.')
    cols = list(saved.get('feature_columns', []))
    if len(cols) != 61 or not cols[:52] or not {'a','b','c','alpha','beta','gamma','Z','unit_cell_volume_A3','cell_volume_per_Z_A3'}.issubset(cols):
        raise ValueError('v28 모델 입력 특징 수 또는 열 구성이 올바르지 않습니다.')
    if not hasattr(saved.get('model'), 'predict'):
        raise ValueError('구조 모델의 예측 메서드가 없습니다.')
    return saved['model'], cols, saved.get('metadata',{})


def v28_build_input(formula, structure, feature_cols):
    comp = Composition(normalize_user_formula(formula))
    if comp.num_atoms <= 0:
        raise ValueError('화학식의 원소 수는 양수여야 합니다.')
    fractions=comp.fractional_composition.get_el_amt_dict()
    f={'elem_'+el:float(q) for el,q in fractions.items()}
    supported={name[5:] for name in feature_cols if name.startswith('elem_')}
    unknown=set(fractions)-supported
    if unknown:
        raise ValueError('학습에서 사용하지 않은 원소가 포함돼 있습니다: '+', '.join(sorted(unknown)))
    latt={k:float(structure[k]) for k in ['a','b','c','alpha','beta','gamma','Z']}
    if not all(np.isfinite(z) for z in latt.values()):
        raise ValueError('격자수치는 유한한 숫자여야 합니다.')
    if not all(latt[x]>0 for x in ['a','b','c','Z']):
        raise ValueError('격자상수·Z는 양수여야 합니다.')
    if not all(0<latt[x]<180 for x in ['alpha','beta','gamma']):
        raise ValueError('격자각은 0도보다 크고 180도보다 작아야 합니다.')
    ca,cb,cg = np.cos(np.deg2rad([latt['alpha'],latt['beta'],latt['gamma']]))
    d=1+2*ca*cb*cg-ca**2-cb**2-cg**2
    if d<=1e-10:
        raise ValueError('격자각으로 계산한 부피가 0 이하입니다. 값들을 다시 확인해주세요.')
    v=latt['a']*latt['b']*latt['c']*float(np.sqrt(d))
    f.update(latt)
    f['unit_cell_volume_A3']=v
    f['cell_volume_per_Z_A3']=v/latt['Z']
    return pd.DataFrame([f],columns=feature_cols).fillna(0.0)


if selected_page == 'structure_pilot':
    st.subheader('🔬 결정구조 기반 시범 예측 · 연구용')
    st.warning('**기본 모델 대체가 아닙니다.** 검토 표시된 171건으로 선별했을 때, 조성·DOI 동시 분리 검증에서 격자 특징을 추가한 모델이 오히려 더 큰 오차를 보였습니다. 예측값을 실제 성능이나 검증된 정확도로 해석하지 마세요.')
    st.caption('학습 자료 449건 중 CIF 처리 완료 · Checked=1 · close match 아님을 만족한 171건으로 만든 시범 모델입니다. CIF 처리 완료가 실제 이온전도도 측정 시료의 구조 일치를 보증하지 않습니다.')
    st.markdown('### 1. 선별 기록 171건의 검증 결과')
    if V28_CV_PATH.is_file():
        v28_cv = pd.read_csv(V28_CV_PATH)
        v28_method=st.radio('검증 분리 방식', ['조성+DOI 연결 분리', '조성 분리'],format_func=lambda x:'조성 + 논문 출처 함께 분리' if 'DOI' in x else '조성만 분리',horizontal=True,key='v28_cv_method')
        v28_view=v28_cv[v28_cv['검증방식']==v28_method].copy()
        if not v28_view.empty:
            v28_view['모델']=v28_view['모델'].map({'COMPOSITION_52':'조성비 52개','LATTICE_61':'조성비 + 격자 61개'})
            st.dataframe(v28_view[['모델','특징개수','전체_기록수','OOF_MAE_log10','OOF_R2']].round(3),hide_index=True,use_container_width=True)
            fig=px.bar(v28_view,x='모델',y='OOF_MAE_log10',text='OOF_MAE_log10',labels={'OOF_MAE_log10':'MAE (log₁₀, 낮을수록 좋음)'},title='선별된 구조 기록의 모델 성능 비교')
            fig.update_traces(texttemplate='%{text:.3f}',textposition='outside')
            fig.update_yaxes(range=[0,float(v28_view['OOF_MAE_log10'].max())*1.2])
            st.plotly_chart(fig,use_container_width=True)
        st.caption('v28 자료에서 같은 171개 기록을 두 모델에 적용했습니다. 5개 폴드 크기는 불균등하며 여러 특징 조합을 탐색한 개발 검증입니다. 독립 테스트 성능이 아닙니다.')
    else:
        st.info('검증 표를 표시하려면 SSE_v28_screened_cv_summary.csv를 app.py와 같은 폴더에 넣어주세요.')

    st.markdown('### 2. 선택한 시료의 구조값으로 시범 예측')
    if not V28_MODEL_PATH.is_file():
        st.info('아직 구조 시범 모델이 설치되지 않았습니다. 배포 ZIP의 Colab 노트북으로 SSE_AI_Structure_Model_v28.joblib을 만든 뒤 복사하세요. 기존 v1·v2 모델은 그대로 사용합니다.')
    else:
        try:
            pilot_model,pilot_cols,pilot_meta=v28_load_pilot_model()
        except Exception as exc:
            st.error('구조 모델 로딩 실패: '+str(exc))
            st.caption('이 모델은 scikit-learn 1.6.1에서 생성·실행하도록 구성했습니다. 컴퓨터와 Colab 버전을 확인하세요.')
        else:
            # Reference structure selection is for demonstration only; manually entered
            # verified structure can be selected for a genuinely independent target.
            mode=st.radio('구조 정보 입력 방식', ['문헌 구조 기록을 참고용으로 선택','실제 시료의 구조값을 직접 입력'],horizontal=False,key='v28_input_mode')
            formula='Li7La3Zr2O12';structure=None
            if mode=='문헌 구조 기록을 참고용으로 선택':
                ftext=st.text_input('화학식 검색 (아래첨자·대소문자 허용)',value='Li7La3Zr2O12',key='v28_formula_ref')
                try:
                    cat,_,_=v27_load_structure_data()
                    compkey=v27_composition_key(ftext)
                    records=cat[cat['composition_key']==compkey]
                except Exception as exc:
                    records=pd.DataFrame();st.error('구조 조회 실패: '+str(exc))
                if not records.empty:
                    choices={int(r.source_row):f"ID {r.ID} · 공간군 {int(r.spacegroup_number)} · CIF {r.cif_status} · DOI {r.doi}" for r in records.itertuples()}
                    record_id=st.selectbox('참고 구조 기록 선택',list(choices),format_func=lambda z:choices[z],key='v28_record')
                    rec=cat[cat['source_row']==record_id].iloc[0]
                    formula=str(rec['formula'])
                    structure={x:float(rec[x]) for x in ['a','b','c','alpha','beta','gamma','Z']}
                    if str(rec['cif_status']).lower()!='done' or str(rec.get('close_match','')).lower()=='yes':
                        st.warning('선택한 기록은 CIF 완료 기준을 충족하지 않거나 close match로 표시되어 있습니다. 구조 출처를 반드시 검증하세요.')
                    st.caption(f"참고 구조: {formula} · 논문 DOI {rec['doi']} · 데이터 ID {rec['ID']}. 기존 개발 자료에서 온 기록이므로 예측값은 **독립 검증이 아닙니다.**")
                else:
                    st.info('현재 개발 자료에서 동일 조성의 구조 기록을 찾지 못했어. 직접 입력 모드를 사용하고 실제 구조 출처를 확인해 주세요.')
            else:
                formula=st.text_input('대상 소재 화학식',value='Li7La3Zr2O12',key='v28_formula_manual')
                st.caption('아래 값은 실제 시료의 XRD/CIF 등 신뢰할 수 있는 측정 자료에서 확인한 값만 입력해 주세요. 임의의 구조값을 생성하지 마세요.')
                fcols=st.columns(4)
                default={'a':13.,'b':13.,'c':13.,'alpha':90.,'beta':90.,'gamma':90.,'Z':8.}
                structure={}
                for i,name in enumerate(['a','b','c','alpha','beta','gamma','Z']):
                    with fcols[i%4]:
                        structure[name]=st.number_input(f'{name} (Å / 도 / Z)',min_value=0.001,max_value=999.0,value=float(default[name]),step=0.1,key='v28_val_'+name)
                source=st.text_input('구조 출처 또는 DOI',placeholder='예: DOI / XRD 파일 식별번호',key='v28_source')
                if not source.strip():
                    st.caption('출처가 확인되지 않은 구조로 모델 성능을 평가하거나 결과를 발표해서는 안 됩니다.')
            approved=st.checkbox('선택한 구조 정보가 **참고용**이며, 예측 정확도가 검증되지 않았고 시료 구조 대응 확인이 필요하다는 점을 이해했습니다.',key='v28_warning_ack')
            if st.button('🧪 연구용 구조 모델로 예측',disabled=structure is None or not approved,key='v28_predict'):
                try:
                    xx=v28_build_input(formula,structure,pilot_cols)
                    log=float(pilot_model.predict(xx)[0]);pred=10**log
                    if not np.isfinite(pred) or pred<=0:
                        raise ValueError('예측값이 유효 범위를 벗어났습니다.')
                    st.success('구조 기반 시범 예측 계산 완료 (독립 검증 아님)')
                    c1,c2=st.columns(2)
                    c1.metric('시범 모델 예측 (S/cm)',f'{pred:.4e}')
                    c2.metric('log₁₀ 예측',f'{log:.4f}')
                    st.caption('참고 구조를 선택한 경우 학습과 겹칠 가능성이 있습니다. 특정 구조의 실제 실험 측정값에 대한 정확도를 보증하지 않습니다.')
                except Exception as exc:
                    st.error('계산 실패: '+str(exc))
    st.markdown('### 3. 앞으로의 검증 과제')
    st.write('구조와 측정 전도도의 시료 일치 여부를 원문에서 확인하고, 동일 논문 외부 검증에서 안정적인 개선이 나타나는지 재검증해야 합니다. 지금은 **화학식 기반 v1·v2 모델 유지**가 권장됩니다.')

# ─────────────────────────────────────────────────────────────────────────────
# v29 배포 전 자가 점검: '파일 있음'과 '계산 통과'를 혼동하지 않음
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == "readiness":
    st.subheader("🧰 앱 점검 · 배포 준비")
    st.write("웹사이트 공개 전에 필요한 파일을 확인하고, 실제 예측과 보고서 생성 기능을 **이 컴퓨터에서** 점검합니다.")
    st.warning("자가 점검 통과는 예측 정확도 검증이나 클라우드 배포 성공을 뜻하지 않습니다. 기존 공식 테스트 결과와 독립적으로 관리합니다.")

    required_files = [
        ("기본 AI 모델", MODEL_PATH, True),
        ("두 번째 AI 모델", MODEL_V2_PATH, True),
        ("구조 시범 모델", V28_MODEL_PATH, False),
        ("학습 참조 데이터", TRAIN_REFERENCE_PATH, True),
        ("내부 검증 참조", VALID_REFERENCE_PATH, False),
        ("공개 원본 데이터 캐시", DATABASE_PATH, False),
        ("구조 기록 목록", V27_CATALOG_PATH, False),
        ("구조 시범 검증 결과", V28_CV_PATH, False),
        ("교차검증 결과", V15_CV_FILES["summary"], False),
    ]
    file_status = pd.DataFrame([
        {"항목": label, "파일": p.name,
         "존재 여부": "있음" if p.is_file() else "없음",
         "설명": "기본 기능에 필요" if essential else "해당 분석 메뉴 사용 시 필요"}
        for label, p, essential in required_files
    ])
    st.markdown("### 1. 실행 파일 준비 상태")
    st.dataframe(file_status, use_container_width=True, hide_index=True)
    if any(essential and not p.is_file() for _, p, essential in required_files):
        st.error("기본 실행 파일 중 일부가 없습니다. 해당 파일을 app.py와 같은 폴더에 추가하세요.")
    else:
        st.success("기본 모델·학습 참조 파일이 존재합니다. 파일이 정상 로딩되는지는 아래 검사를 실행해야 합니다.")

    st.markdown("### 2. 핵심 기능 계산 자가 검사")
    st.caption("검사는 버튼을 눌렀을 때만 수행합니다. 현재 모델을 재학습하거나 저장 파일을 변경하지 않습니다.")
    if st.button("🔍 예측·PDF·구조 모델 작동 검사", key="v29_run_checks", use_container_width=True):
        checks = []
        def log_check(name, passed, detail):
            checks.append({"점검 항목": name, "결과": "통과" if passed else "확인 필요", "설명": detail})
        try:
            x = predict_one("Li₆PS₅Cl")
            y = predict_one("li6ps5cl")
            correct = (np.isfinite(x["conductivity_S_per_cm"]) and
                       np.isclose(x["conductivity_S_per_cm"], y["conductivity_S_per_cm"], rtol=1e-9))
            log_check("v1 예측 및 화학식 자동 보정", correct,
                      f"정규 입력·소문자 입력 모두 {x['conductivity_S_per_cm']:.4e} S/cm")
        except Exception as exc:
            log_check("v1 예측 및 화학식 자동 보정", False, str(exc))
        try:
            if not MODEL_V2_PATH.is_file():
                raise FileNotFoundError("두 번째 모델 파일이 없습니다")
            mod2, cols2 = load_model_v2()
            p2 = float(mod2.predict(make_input_v2("Li6PS5Cl", cols2))[0])
            log_check("v2 예측", np.isfinite(p2), f"log₁₀(S/cm)={p2:.4f}")
        except Exception as exc:
            log_check("v2 예측", False, str(exc))
        try:
            if not TRAIN_REFERENCE_PATH.is_file():
                raise FileNotFoundError("SSE_train_reference.csv 파일이 없습니다")
            df = pd.read_csv(TRAIN_REFERENCE_PATH)
            log_check("학습 참조 파일", len(df)>0, f"CSV {len(df)}건")
        except Exception as exc:
            log_check("학습 참조 파일", False, str(exc))
        try:
            from reportlab.pdfgen import canvas
            f = io.BytesIO()
            page = canvas.Canvas(f)
            page.drawString(30, 760, "SSE-AI Lab PDF engine smoke test")
            page.save()
            payload = f.getvalue()
            log_check("PDF 엔진", payload.startswith(b"%PDF") and len(payload)>100,
                      "PDF 파일 작성 가능 (한글 보고서 전체 렌더링과는 별도)")
        except Exception as exc:
            log_check("PDF 엔진", False, str(exc))
        try:
            if not V28_MODEL_PATH.is_file():
                raise FileNotFoundError("시범 모델 파일이 없습니다 (구조 메뉴 전용)")
            smodel, scols, _meta = v28_load_pilot_model()
            log_check("구조 시범 모델 로딩", len(scols)==61 and hasattr(smodel, 'predict'),
                      "61개 특징 모델 로딩 확인 (예측 정확도와 별도)")
        except Exception as exc:
            log_check("구조 시범 모델 로딩", False, str(exc))
        st.session_state["v29_self_check_results"] = checks
    if "v29_self_check_results" in st.session_state:
        results = pd.DataFrame(st.session_state["v29_self_check_results"])
        st.dataframe(results, hide_index=True, use_container_width=True)
        n_ok = sum(x == "통과" for x in results["결과"])
        st.write(f"이번 실행의 자가 검사 결과: **{n_ok}/{len(results)}항목 통과**")
        st.download_button("📥 자가 검사 결과 CSV 저장",
            data=results.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"SSE_AI_Lab_{APP_FILE_TAG}_self_check.csv", mime="text/csv")

    st.markdown("### 3. 발표 및 웹 공개 전 확인 사항")
    st.markdown("""
- **AI 예측과 두 모델 비교:** `Li6PS5Cl` 결과 확인, 아래첨자·대소문자 입력 검사
- **소재 검색·추천:** 출처 DOI와 실험값/예측값 구분, 긴 화학식의 가독성 확인
- **시뮬레이션·PDF:** 21개 조성 결과와 CSV, 한국어 PDF 다운로드 열기
- **데이터 품질·교차검증:** 동일 조성 측정값과 개발용 검증 결과의 의미 구분
- **구조 시범 예측:** 구조 출처 확인, 실험용 모델로만 사용
- **웹 배포:** 별도 서버에서 모델/CSV/한글 글꼴 설치 및 모든 메뉴 재검사
""")
    st.info("현재 확인한 모델의 개발용 수치가 좋아도 독립 테스트 정확도를 보증하지 않습니다. "
            "기존 공식 테스트(v1: MAE 1.522, R² -0.084)와 구조 모델의 탐색적 교차검증을 구분해서 발표하세요.")
    checklist = f"""SSE-AI Lab {APP_VERSION} 배포 전 점검표
□ 예측/검색/추천 정상
□ 대소문자·아래첨자 입력 정상
□ 21개 조성 시뮬레이션과 CSV 저장
□ 통합 분석과 한국어 PDF 다운로드
□ 반복 측정 기록/출처 확인
□ 교차검증 요약과 모델 차이 해석
□ 구조 시범 모델은 연구용임을 확인
□ 다른 컴퓨터 또는 클라우드에서 전체 화면 테스트
□ 개인정보·비공개 모델 데이터의 공개 여부 검토
"""
    st.download_button("📥 배포 전 점검표 TXT 저장", data=checklist.encode("utf-8-sig"),
                       file_name=f"SSE_AI_Lab_{APP_FILE_TAG}_deployment_checklist.txt", mime="text/plain")


st.divider()
st.caption("SSE-AI Lab v29.3 · 연구용 참고 도구 | 예측값은 실험값이 아니며 예측 정확도가 보장되지 않습니다.")
