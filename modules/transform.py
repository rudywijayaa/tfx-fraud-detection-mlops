
import tensorflow_transform as tft


NUMERIC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
LABEL_KEY = "Class"


def _clean_key(key: str) -> str:
    return key.replace('"', '')


def transformed_name(key: str) -> str:
    return f"{_clean_key(key)}_xf"


def preprocessing_fn(inputs: dict) -> dict:
    outputs = {}
    cleaned_inputs = {_clean_key(k): v for k, v in inputs.items()}

    for feature_name in NUMERIC_FEATURES:
        outputs[transformed_name(feature_name)] = tft.scale_to_z_score(
            cleaned_inputs[feature_name]
        )

    outputs[transformed_name(LABEL_KEY)] = cleaned_inputs[LABEL_KEY]
    return outputs
