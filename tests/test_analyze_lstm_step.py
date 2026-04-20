from services.analyze_service import AnalyzeService


def main():
    service = AnalyzeService(
        mode="full",
        risk_method="weighted",
        dataset_key="cmapss_lstm",
    )

    feature_dim = service.pdm_service._meta["feature_dim"]

    sequence = [
        [0.1] * feature_dim,
        [0.2] * feature_dim,
        [0.3] * feature_dim,
        [0.4] * feature_dim,
    ]

    feature_names = [f"f{i}" for i in range(feature_dim)]

    result = service.run_lstm(
        sequence=sequence,
        feature_names=feature_names,
        asset_id="ENGINE-001",
        dataset_key="cmapss_lstm",
    )

    print("validation_text:")
    print(result["validation_text"])
    print()

    print("summary_text:")
    print(result["summary_text"])
    print()

    print("explanation_text:")
    print(result["explanation_text"])
    print()

    print("alert_text:", result["alert_text"])
    print()

    print("report_markdown:")
    print(result["report_markdown"][:1000])
    print()

    raw_result = result["raw_result"]
    print("raw_result keys:", raw_result.keys() if raw_result else None)


if __name__ == "__main__":
    main()