# exp004_xgboost 코드 스냅샷

이 폴더는 exp004_xgboost 제출 파일을 만들 때 사용한 코드와 설정을 보관한다.

`src/`는 다음 실험에서 계속 수정될 수 있으므로, exp004_xgboost 재현용 코드는 이 폴더의 스냅샷을 기준으로 확인한다.

## 파일

- `code/make_features.py`: LDAPS/GFS baseline 피처 생성
- `code/train.py`: XGBoost 학습 및 2024년 내부 검증
- `code/predict.py`: 2025년 예측 및 제출 파일 생성
- `code/metrics.py`: NMAE, clip 등 공통 함수
- `exp004_xgboost.yaml`: exp004_xgboost 설정

## 실행 순서

프로젝트 루트에서 실행한다.

```bash
python experiments/exp004_xgboost/code/make_features.py --config experiments/exp004_xgboost/exp004_xgboost.yaml
python experiments/exp004_xgboost/code/train.py --config experiments/exp004_xgboost/exp004_xgboost.yaml
python experiments/exp004_xgboost/code/predict.py --config experiments/exp004_xgboost/exp004_xgboost.yaml
```

주의: 설정 파일 안의 출력 경로는 `outputs/exp004_xgboost`, `submissions/exp004_xgboost_submission.csv`를 가리킨다.

## 제출 파일

대회 제출 파일은 CSV 형식이다.

- 제출 파일: `submissions/exp004_xgboost_submission.csv`
- 행 수: `8760`
- 컬럼: `forecast_id`, `forecast_kst_dtm`, `kpx_group_1`, `kpx_group_2`, `kpx_group_3`

## 코드 해시

```text
9ef4ba3b0266589a304ecb5a3ba8ea201d9b6dff566255cd56419ccca84bf235  code/make_features.py
24df115c8096b76b7a09abbdc6398adb621cacf361cb1fe6498096ffcc0c3160  code/metrics.py
0657cd19b0c3057bca20f3505de3cfe7059fd48fc745a0396489081af5c54ecf  code/predict.py
a560ae5f4c72dd9790ae5d078e4fe0816b1ee353cbd7d0139709bc81d335db3c  code/train.py
65b967203b935d4674fd1b640baae800b249b03fbc8468e3c2505049ed05ecb1  exp004_xgboost.yaml
```
