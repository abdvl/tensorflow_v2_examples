import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv

# Simple dataset analysis
dataset = pd.read_csv("data/pollution-small.csv")
print(dataset.shape)

training_data = dataset[:1600]
print(training_data.describe())

test_set = dataset[1600:]
print(test_set.describe())

# Generate training data statistics
train_stats = tfdv.generate_statistics_from_dataframe(dataframe=dataset)
schema = tfdv.infer_schema(statistics=train_stats)
print(tfdv.display_schema(schema))


test_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_set)

# Compare test statistics with the Schema
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
# Displaying all detected anomalies
# Integer larger than 10
# STRING type when expected INT type
# FLOAT type when expected INT type
# Integer smaller than 0
print(tfdv.display_anomalies(anomalies))

# New data WITH anomalies
test_set_copy = test_set.copy()
test_set_copy.drop("soot", axis=1, inplace=True)


# Statistics based on data with anomalies
test_set_copy_stats = tfdv.generate_statistics_from_dataframe(dataframe=test_set_copy)
anomalies_new = tfdv.validate_statistics(statistics=test_set_copy_stats, schema=schema)
print(tfdv.display_anomalies(anomalies_new))

# Prepare the schema for Serving,environment
schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")

tfdv.get_feature(schema, "soot").not_in_environment.append("SERVING")

serving_env_anomalies = tfdv.validate_statistics(test_set_copy_stats, schema, environment="SERVING")
print(tfdv.display_anomalies(serving_env_anomalies))

# Freezing the schema
tfdv.write_schema_text(schema=schema,output_path='pollution_schema.pbtext')