import json
import os

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult


NUMERIC_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
LABEL_KEY = "Class"


def transformed_name(key: str) -> str:
    return f"{key.replace(chr(34), '')}_xf"


def _input_fn(
    file_pattern: str,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 64,
) -> tf.data.Dataset:
    actual_files = tf.io.gfile.glob(file_pattern)
    if actual_files and tf.io.gfile.isdir(actual_files[0]):
        actual_files = tf.io.gfile.glob(os.path.join(file_pattern, "*"))

    feature_spec = tf_transform_output.transformed_feature_spec().copy()

    def parse_tfrecord(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_spec)
        label = parsed_features.pop(transformed_name(LABEL_KEY))
        return parsed_features, label

    dataset = tf.data.TFRecordDataset(actual_files, compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_keras_model(hp: kt.HyperParameters) -> tf.keras.Model:
    inputs = [
        tf.keras.Input(shape=(1,), name=transformed_name(feature_name))
        for feature_name in NUMERIC_FEATURES
    ]
    x = tf.keras.layers.concatenate(inputs)
    x = tf.keras.layers.Dense(
        units=hp.Int("hidden_units", min_value=32, max_value=128, step=32),
        activation="relu",
    )(x)
    x = tf.keras.layers.Dropout(hp.Float("dropout", 0.1, 0.4, step=0.1))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4])
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)
    tuner = kt.RandomSearch(
        hypermodel=_build_keras_model,
        objective=kt.Objective("val_pr_auc", direction="max"),
        max_trials=4,
        directory=fn_args.working_dir,
        project_name="fraud_detection",
        overwrite=False,
    )
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 3,
            "class_weight": {0: 1.0, 1: 578.0},
        },
    )