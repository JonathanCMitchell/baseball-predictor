import tempfile
import urllib.request
from IPython import embed

train_file = 'number_as_position_2015.csv'
test_file = 'number_as_position_2016.csv'

import pandas as pd

#COLUMNS = ["Rk","Name","Age","Tm","G","PA","AB","R","H","2B","3B","HR","RBI","SB","CS","BB","SO","BA",
           #"OBP","SLG","OPS","OPS+","TB","GDP","HBP","SH","SF","IBB","Pos"]

COLUMNS = ["Name", "AB", "HR", "RBI", "SLG", "Pos"]

# based off of HR if someone has hit more than 30 RBIs
df_train = pd.read_csv(train_file, names=COLUMNS, usecols=[1,6,11,12,19,28], skipinitialspace=True, skiprows=1)
df_test = pd.read_csv(test_file, names=COLUMNS, usecols=[1,6,11,12,19,28], skipinitialspace=True, skiprows=1)


#LABEL_COLUMN = ">30 RBIs"
#df_train[LABEL_COLUMN] = (df_train["RBI"].apply(lambda x: x > 30)).astype(int)
#df_test[LABEL_COLUMN] = (df_test["RBI"].apply(lambda x: x > 30)).astype(int)

#df_train["outfield"] = (df_train["Pos"].apply(lambda x: x === "OF")).astype(int)
LABEL_COLUMN = "Position"
df_train[LABEL_COLUMN] = (df_train["Pos"].apply(lambda x: x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["Pos"].apply(lambda x: x)).astype(int)

CONTINUOUS_COLUMNS = ["AB", "HR", "SLG"]

import tensorflow as tf

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}

  # Merges the two dictionaries into one.
  # feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  feature_cols = continuous_cols
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

hr = tf.contrib.layers.real_valued_column("HR")
slg = tf.contrib.layers.real_valued_column("SLG")
ab = tf.contrib.layers.real_valued_column("AB")


hr_buckets = tf.contrib.layers.bucketized_column(hr, boundaries=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
slg_buckets = tf.contrib.layers.bucketized_column(slg, boundaries=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ab_buckets = tf.contrib.layers.bucketized_column(ab, boundaries=[1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 550, 650])

hr_x_slg = tf.contrib.layers.crossed_column([hr_buckets, slg_buckets], hash_bucket_size=int(10000))
slg_x_ab = tf.contrib.layers.crossed_column([ab_buckets, slg_buckets], hash_bucket_size=int(1000))


model_dir = tempfile.mkdtemp()

# n_classes is used to determine how many target classes there are starting from 0, this is binary by default
model = tf.contrib.learn.LinearClassifier(feature_columns=[hr, slg_x_ab], model_dir=model_dir, n_classes = 8)

model.fit(input_fn=train_input_fn, steps=200)



results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print ("{}: {}".format(key, results[key]))
