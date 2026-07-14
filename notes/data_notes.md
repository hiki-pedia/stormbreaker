# 데이터 메모

## 원본 파일

- `train/ldaps_train.csv`: 학습용 LDAPS 기상예보 데이터.
- `train/gfs_train.csv`: 학습용 GFS 기상예보 데이터.
- `train/train_labels.csv`: KPX 그룹별 시간 단위 발전량 정답.
- `train/scada_vestas_train.csv`: VESTAS 터빈 SCADA 데이터.
- `train/scada_unison_train.csv`: UNISON 터빈 SCADA 데이터.
- `test/ldaps_test.csv`: 2025년 추론용 LDAPS 기상예보 데이터.
- `test/gfs_test.csv`: 2025년 추론용 GFS 기상예보 데이터.
- `sample_submission.csv`: 제출 파일 양식.
- `info.xlsx`: 터빈 메타데이터, 위치, 그룹, 설비용량 정보.

## 데이터 제공 범위

- 학습용 기상 데이터와 정답 데이터는 `2022-01-01 01:00:00`부터 `2025-01-01 00:00:00`까지 제공된다.
- 평가용 기상 데이터와 제출 행은 `2025-01-01 01:00:00`부터 `2026-01-01 00:00:00`까지 제공된다.
- `kpx_group_3` 정답은 2023년부터 제공된다.
- VESTAS SCADA는 2022-2024년 구간이 제공된다.
- UNISON SCADA는 2023-2024년 구간이 제공된다.

## 데이터 누수 주의사항

- 어떤 데이터를 사용할 수 있는지 판단할 때는 예보가 공개되는 시점을 기준으로 생각한다.
- 2025년 실제 발전량은 사용하지 않는다.
- 2025년 SCADA 또는 실제 터빈 관측값은 사용하지 않는다.
- 미래 관측값이 있어야 만들 수 있는 피처는 사용하지 않는다.

## 알려진 결측

- `kpx_group_3`는 2022년 정답이 없다.
- `kpx_group_1`, `kpx_group_2`는 짧은 정답 결측 구간이 일부 있다.
- `test/ldaps_test.csv`는 몇몇 시각과 변수에서 작은 결측 구간이 있다.
- UNISON SCADA에는 센서값 내부 결측이 있다.
