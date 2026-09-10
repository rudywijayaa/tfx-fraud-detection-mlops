
import json
import os

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
LABEL_KEY = "Class"


def _clean_key(key: str) -> str:
    return key.replace('"', '')


def transformed_name(key: str) -> str:
    return f"{_clean_key(key)}_xf"


def _build_keras_model(hyperparameters=None) -> tf.keras.Model:
    hyperparameters = hyperparameters or {}
    hidden_units = int(hyperparameters.get("hidden_units", 64))
    dropout = float(hyperparameters.get("dropout", 0.2))
    learning_rate = float(hyperparameters.get("learning_rate", 0.001))

    inputs = [
        tf.keras.Input(shape=(1,), name=transformed_name(feature_name))
        for feature_name in NUMERIC_FEATURES
    ]
    x = tf.keras.layers.concatenate(inputs)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(max(hidden_units // 2, 16), activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def _input_fn(
    file_pattern: str,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 64,
) -> tf.data.Dataset:
    actual_files = tf.io.gfile.glob(file_pattern)
    if actual_files and tf.io.gfile.isdir(actual_files[0]):
        actual_files = tf.io.gfile.glob(os.path.join(file_pattern, "*"))
    raw_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    def parse_tfrecord(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, raw_feature_spec)
        label = parsed_features.pop(transformed_name(LABEL_KEY))
        return parsed_features, label

    dataset = tf.data.TFRecordDataset(actual_files, compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat fungsi serving signature yang memproses raw TF.Examples."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        # Filter fitur agar hanya menyertakan yang dibutuhkan oleh model
        model_inputs = {
            transformed_name(feat): transformed_features[transformed_name(feat)]
            for feat in NUMERIC_FEATURES
        }
        return model(model_inputs)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    if isinstance(fn_args.hyperparameters, str):
        hyperparameters = json.loads(fn_args.hyperparameters)
    elif fn_args.hyperparameters:
        hyperparameters = fn_args.hyperparameters
    else:
        hyperparameters = {}

    model = _build_keras_model(hyperparameters)
    
    # Penentuan train/eval steps berpatokan pada fn_args
    train_steps = fn_args.train_steps or 1000
    eval_steps = fn_args.eval_steps or 200

    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        class_weight={0: 1.0, 1: 578.0},
        epochs=5,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
