import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam  # apache beam

print(tf.__version__)

# from __future__ import print_function
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema

# read and prepare data
dataset = pd.read_csv("data/pollution-small.csv")
print(dataset.head())

features = dataset.drop("Date", axis=1)
print(features.head())

# Converting the dataset from dataframe to list of Python dictionaries
dict_features = list(features.to_dict("index").values())

print(dict_features[:2])

# Defining the dataset metadata
data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.schema_utils.schema_from_feature_spec({
        "no2": tf.io.FixedLenFeature([], tf.float32),
        "so2": tf.io.FixedLenFeature([], tf.float32),
        "pm10": tf.io.FixedLenFeature([], tf.float32),
        "soot": tf.io.FixedLenFeature([], tf.float32),
    })
)
print(data_metadata)

# The preprocessing function

def preprocessing_fn(inputs):
    no2 = inputs['no2']
    pm10 = inputs['pm10']
    so2 = inputs['so2']
    soot = inputs['soot']

    no2_normalized = no2 - tft.mean(no2)
    so2_normalized = so2 - tft.mean(so2)

    pm10_normalized = tft.scale_to_0_1(pm10)
    soot_normalized = tft.scale_by_min_max(soot)

    return {
        "no2_normalized": no2_normalized,
        "so2_normalized": so2_normalized,
        "pm10_normalized": pm10_normalized,
        "soot_normalized": soot_normalized
    }


def data_transform():
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (
                    (dict_features, data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

    transformed_data, transformed_metadata = transformed_dataset

    for i in range(len(transformed_data)):
        pass
        print("Raw: ", dict_features[i])
        print("Transformed:", transformed_data[i])

data_transform()