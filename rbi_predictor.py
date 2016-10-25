import tempfile

train_file = 'number_as_position_2015.csv'
test_file = 'number_as_position_2016.csv'

import pandas as pd

# COLUMNS_AVAILABLE = ["Name", "Age", "AB", "R", "H", "2B", "3B" , "HR" , "RBI", "SB", "BB", "SO", "BA", "OBP", "Pos"]
'''
Here we are specifying which columns we want to use from the CSV, since we are predicting the amount of RBIs
for a player we need at least that column.  Also we need to use the other columns we would like to use in predicting
if a player will hit more or less than 65 RBIs
'''

COLUMNS = ["Age", "RBI"]

'''
usecols specifies which columns we want to use, 1 for the Age, and 8 for the RBI since that is where they are indexed.
'''

df_train = pd.read_csv(train_file, names=COLUMNS, usecols=[1,8], skipinitialspace=True, skiprows=1)
df_test = pd.read_csv(test_file, names=COLUMNS, usecols=[1,8], skipinitialspace=True, skiprows=1)


LABEL_COLUMN = "RBI"
df_train[LABEL_COLUMN] = (df_train["RBI"].apply(lambda x: x > 65)).astype(int)
df_test[LABEL_COLUMN] = (df_test["RBI"].apply(lambda x: x < 65)).astype(int)

CONTINUOUS_COLUMNS = ["Age"]

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

age = tf.contrib.layers.real_valued_column("Age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 20, 22, 25, 27, 30, 32, 35, 38, 40])

model_dir = tempfile.mkdtemp()

model = tf.contrib.learn.LinearClassifier(feature_columns=[age_buckets], model_dir=model_dir, n_classes = 2)

model.fit(input_fn=train_input_fn, steps=200)

results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print ("{}: {}".format(key, results[key]))
