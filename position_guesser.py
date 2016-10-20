import tempfile

train_file = 'number_as_position_2015.csv'
test_file = 'number_as_position_2016.csv'

import pandas as pd

# COLUMNS_AVAILABLE = ["Name","Age","AB","R","H","2B","3B","HR","RBI","SB","BB","SO","BA","OBP","Pos" ]

COLUMNS = ["Name", "AB", "HR", "RBI", "Pos"]

df_train = pd.read_csv(train_file, names=COLUMNS, usecols=[0,2,7,8,14], skipinitialspace=True, skiprows=1)
df_test = pd.read_csv(test_file, names=COLUMNS, usecols=[0,2,7,8,14], skipinitialspace=True, skiprows=1)


LABEL_COLUMN = "Position"
df_train[LABEL_COLUMN] = (df_train["Pos"].apply(lambda x: x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["Pos"].apply(lambda x: x)).astype(int)

CONTINUOUS_COLUMNS = ["AB", "HR", "RBI"]


import tensorflow as tf

def input_fn(df):
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}

  feature_cols = continuous_cols
  label = tf.constant(df[LABEL_COLUMN].values)
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

hr = tf.contrib.layers.real_valued_column("HR")
ab = tf.contrib.layers.real_valued_column("AB")

hr_buckets = tf.contrib.layers.bucketized_column(hr, boundaries=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ab_buckets = tf.contrib.layers.bucketized_column(ab, boundaries=[1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 550, 650])

model_dir = tempfile.mkdtemp()

model = tf.contrib.learn.LinearClassifier(feature_columns=[hr, ab], model_dir=model_dir, n_classes = 8)

model.fit(input_fn=train_input_fn, steps=200)

results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print ("{}: {}".format(key, results[key]))
