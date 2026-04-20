from services.input_validation_service import InputValidationService
from services.schemas import LSTMSequenceInput


def print_divider(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_existing_ai4i_validation():
    print_divider("1) 기존 AI4I scalar validation 테스트")

    svc = InputValidationService()

    payload = {
        "air_temperature_k": 298.5,
        "process_temperature_k": 309.1,
        "rotational_speed_rpm": 1510.0,
        "torque_nm": 36.0,
        "tool_wear_min": 42.0,
    }

    result = svc.validate(payload)

    print("is_valid:", result.is_valid)
    print("normalized_payload:", result.normalized_payload)
    print("issues:", result.issues)

    assert result.is_valid is True
    assert len(result.issues) == 0
    assert "air_temperature_k" in result.normalized_payload


def test_valid_lstm_sequence():
    print_divider("2) 정상 LSTM sequence validation 테스트")

    svc = InputValidationService()

    payload = LSTMSequenceInput(
        sequence=[
            [0.10, 0.20, 0.30],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
        ],
        feature_names=["f1", "f2", "f3"],
        asset_id="ENGINE-001",
        dataset_key="cmapss_lstm",
    )

    result = svc.validate_lstm_sequence_payload(payload)

    print("is_valid:", result.is_valid)
    print("timesteps:", result.timesteps)
    print("feature_dim:", result.feature_dim)
    print("feature_names:", result.feature_names)
    print("issues:", result.issues)

    assert result.is_valid is True
    assert result.timesteps == 3
    assert result.feature_dim == 3
    assert len(result.issues) == 0


def test_invalid_lstm_non_numeric():
    print_divider("3) 비정상 LSTM sequence - 숫자 변환 실패 테스트")

    svc = InputValidationService()

    payload = LSTMSequenceInput(
        sequence=[
            [0.10, 0.20, 0.30],
            [0.11, "bad", 0.31],
            [0.12, 0.22, 0.32],
        ],
        feature_names=["f1", "f2", "f3"],
        asset_id="ENGINE-002",
        dataset_key="cmapss_lstm",
    )

    result = svc.validate_lstm_sequence_payload(payload)

    print("is_valid:", result.is_valid)
    print("issues:")
    for issue in result.issues:
        print(f" - [{issue.level}] {issue.field}: {issue.message}")

    assert result.is_valid is False
    assert any("숫자형으로 변환할 수 없습니다" in issue.message for issue in result.issues)


def test_invalid_lstm_inconsistent_row_length():
    print_divider("4) 비정상 LSTM sequence - row 길이 불일치 테스트")

    svc = InputValidationService()

    payload = LSTMSequenceInput(
        sequence=[
            [0.10, 0.20, 0.30],
            [0.11, 0.21],
            [0.12, 0.22, 0.32],
        ],
        feature_names=["f1", "f2", "f3"],
        asset_id="ENGINE-003",
        dataset_key="cmapss_lstm",
    )

    result = svc.validate_lstm_sequence_payload(payload)

    print("is_valid:", result.is_valid)
    print("issues:")
    for issue in result.issues:
        print(f" - [{issue.level}] {issue.field}: {issue.message}")

    assert result.is_valid is False
    assert any("feature 수가 일치하지 않습니다" in issue.message for issue in result.issues)


def test_invalid_lstm_feature_names_mismatch():
    print_divider("5) 비정상 LSTM sequence - feature_names 길이 불일치 테스트")

    svc = InputValidationService()

    payload = LSTMSequenceInput(
        sequence=[
            [0.10, 0.20, 0.30],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
        ],
        feature_names=["f1", "f2"],  # intentionally wrong
        asset_id="ENGINE-004",
        dataset_key="cmapss_lstm",
    )

    result = svc.validate_lstm_sequence_payload(payload)

    print("is_valid:", result.is_valid)
    print("issues:")
    for issue in result.issues:
        print(f" - [{issue.level}] {issue.field}: {issue.message}")

    assert result.is_valid is False
    assert any("feature_names 길이가 feature 수와 일치하지 않습니다" in issue.message for issue in result.issues)


def test_invalid_lstm_dataset_key():
    print_divider("6) 비정상 LSTM sequence - dataset_key 오류 테스트")

    svc = InputValidationService()

    payload = LSTMSequenceInput(
        sequence=[
            [0.10, 0.20, 0.30],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
        ],
        feature_names=["f1", "f2", "f3"],
        asset_id="ENGINE-005",
        dataset_key="unknown_lstm",
    )

    result = svc.validate_lstm_sequence_payload(payload)

    print("is_valid:", result.is_valid)
    print("issues:")
    for issue in result.issues:
        print(f" - [{issue.level}] {issue.field}: {issue.message}")

    assert result.is_valid is False
    assert any("지원하지 않는 LSTM dataset_key" in issue.message for issue in result.issues)


if __name__ == "__main__":
    test_existing_ai4i_validation()
    test_valid_lstm_sequence()
    test_invalid_lstm_non_numeric()
    test_invalid_lstm_inconsistent_row_length()
    test_invalid_lstm_feature_names_mismatch()
    test_invalid_lstm_dataset_key()

    print("\n모든 validation 테스트가 끝났습니다.")