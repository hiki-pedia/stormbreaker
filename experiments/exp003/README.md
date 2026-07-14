# exp003 코드 스냅샷

이 폴더는 exp003 제출 파일을 만들 때 사용한 코드와 설정을 보관한다.

`src/`는 다음 실험에서 계속 수정될 수 있으므로, exp003 재현용 코드는 이 폴더의 스냅샷을 기준으로 확인한다.

## 파일

- `code/make_features.py`: LDAPS/GFS 피처 생성, 풍력 물리 피처 포함
- `code/train.py`: LightGBM 학습 및 2024년 내부 검증
- `code/predict.py`: 2025년 예측 및 제출 파일 생성
- `code/metrics.py`: NMAE, clip 등 공통 함수
- `exp003_physics.yaml`: exp003 설정

## 실행 순서

프로젝트 루트에서 실행한다.

```bash
python experiments/exp003/code/make_features.py --config experiments/exp003/exp003_physics.yaml
python experiments/exp003/code/train.py --config experiments/exp003/exp003_physics.yaml
python experiments/exp003/code/predict.py --config experiments/exp003/exp003_physics.yaml
```

주의: 설정 파일 안의 출력 경로는 `outputs/exp003`, `submissions/exp003_submission.csv`를 가리킨다.

## 제출 파일

대회 제출 파일은 CSV 형식이다.

- 제출 파일: `submissions/exp003_submission.csv`
- 행 수: `8760`
- 컬럼: `forecast_id`, `forecast_kst_dtm`, `kpx_group_1`, `kpx_group_2`, `kpx_group_3`

## 코드 해시

```text
9ef4ba3b0266589a304ecb5a3ba8ea201d9b6dff566255cd56419ccca84bf235  code/make_features.py
24df115c8096b76b7a09abbdc6398adb621cacf361cb1fe6498096ffcc0c3160  code/metrics.py
0657cd19b0c3057bca20f3505de3cfe7059fd48fc745a0396489081af5c54ecf  code/predict.py
f0a7c90382f848ec5488421c24e332038810ca862e38e1f8934474bc7222e6c5  code/train.py
37e4b657da16127c79c4a235535032b7f6ca3d796c858d56ad1c7c72f89ae9a3  exp003_physics.yaml
```

