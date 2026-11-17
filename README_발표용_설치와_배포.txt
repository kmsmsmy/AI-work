SSE-AI Lab — 2차 중간발표 시연용
================================

이 앱은 기존 v29.1의 예측·검증 계산을 수정하지 않고, 2차 중간발표에서 보여 줄 메뉴만 정리한 발표용 화면입니다.

[바로 실행 — 내 컴퓨터]
1) 기존 SSE_AI_Lab_Starter 폴더 전체를 복사해서 별도 폴더(SSE_AI_Lab_2차발표)로 만드세요.
2) 이 ZIP의 app.py 및 동봉 CSV 7개를 별도 폴더에 복사하세요.
3) 기존 폴더에 있는 다음 파일을 별도 폴더에도 남겨두세요.
   필수: SSE_AI_Model_v1.joblib, SSE_AI_Model_v2.joblib,
         SSE_train_reference.csv, SSE_valid_reference.csv,
         OBELiX_train.csv (있다면; 없으면 온라인에서 공식 학습 데이터 다운로드 시도)
4) CMD에서 발표용 폴더로 이동 후 다음을 실행합니다.
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
5) http://localhost:8501

[웹사이트로 배포 — Streamlit Community Cloud]
- ZIP 자체를 Streamlit에 직접 업로드하는 게 아니라, GitHub 저장소 최상위에 압축 해제한 파일을 올립니다.
- app.py / requirements.txt / .streamlit/config.toml / 제공된 CSV 7개 외에도
  PC에 있는 두 모델 joblib 및 학습/검증 CSV 2개를 같은 폴더에 업로드해야 합니다.
- OBELiX_train.csv를 함께 업로드할 것을 권장합니다. 없으면 최초 데이터 품질 검색 시
  공식 저장소의 학습 CSV를 다운로드하며, 인터넷 연결이 되지 않으면 해당 메뉴가 실패할 수 있습니다.
- 처음 생성 시 Python 3.12를 선택합니다. 학습 당시 scikit-learn 버전 1.6.1을 맞춥니다.
- Main file path: app.py
- 공개 GitHub 저장소에 모델 파일을 올리면 누구나 다운로드할 수 있습니다.
  공개 여부와 외부 데이터 재배포 조건을 먼저 확인하세요. 공식 테스트셋은 추가하지 마세요.

[발표 시연 순서]
1. 시작 안내
2. AI 예측: Li6PS5Cl
3. 조성 시뮬레이션: Br 50%, 21개 조성 그래프
4. 데이터 품질: 분석 실행 -> Li6PS5Cl 측정값 3개 / 약 62배 차이
5. 내부 검증: 354건 학습, 95건 검증 / 별도 공식 테스트 109건
6. 교차검증: 조성과 DOI 동시 분리 / A-B-C 비교 / MAE 1.289, 1.397, 1.297
7. 결정구조: Li7La3Zr2O12 문헌 5건 / 공간군 2종

[주의]
- 이번 앱은 중간발표에 해당하는 메뉴 9개만 보여줍니다. 나머지 기능은 원래 v29.1에 유지됩니다.
- 계산 코드를 변경하지 않았습니다. AI 추정값을 실험 측정값으로 제시하지 마세요.
- 모델 2개 및 학습/검증 CSV 2개는 이 ZIP에 포함되지 않았습니다.
- 실제 웹사이트 배포와 외부 접속 검사는 아직 수행되지 않았습니다.

데이터 출처: NRC-Mila OBELiX (CC BY 4.0), https://github.com/NRC-Mila/OBELiX
