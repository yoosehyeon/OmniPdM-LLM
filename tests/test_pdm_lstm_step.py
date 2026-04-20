from services.pdm_service import PdmService


def main():
    # 실제 체크포인트가 있어야 합니다.
    service = PdmService(mode="full", dataset_key="cmapss_lstm")

    # feature_dim은 실제 모델 학습 차원과 반드시 일치해야 합니다.
    # 아래는 예시입니다. shape mismatch가 나면 feature 수를 맞춰야 합니다.
    sequence = [
        [0.1] * service._meta["feature_dim"],
        [0.2] * service._meta["feature_dim"],
        [0.3] * service._meta["feature_dim"],
        [0.4] * service._meta["feature_dim"],
    ]

    feature_names = [f"f{i}" for i in range(service._meta["feature_dim"])]

    result = service.predict_lstm_sequence(
        sequence=sequence,
        feature_names=feature_names,
    )

    print("dataset_key:", result.dataset_key)
    print("task_type:", result.task_type)
    print("model_name:", result.model_name)
    print("predicted_label:", result.predicted_label)
    print("failure_probability:", result.failure_probability)
    print("anomaly_score:", result.anomaly_score)
    print("rul_norm:", result.rul_norm)
    print("top_contributors:", result.top_contributors)
    print("raw_output:", result.raw_output)


if __name__ == "__main__":
    main()