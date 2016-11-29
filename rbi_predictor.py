import tempfile
train_file = 'number_as_position_2015.csv'
test_file = 'number_as_position_2016.csv'

import pandas as pd

# COLUMNS_AVAILABLE = ["Name", "Age", "AB", "R", "H", "2B", "3B" , "HR" , "RBI", "SB", "BB", "SO", "BA", "OBP", "Pos"]
'''
Here we are specifying which columns we want to use from the CSV, since we are predicting the amount of RBIs
for a player we need at least that column.  Also we need to use the other columns we would like to use in predicting
if a player will hit more or less than 65 RBIs, here we'll use the amount of stolen bases
'''

COLUMNS = ["RBI", "SB"]

'''
usecols specifies which columns we want to use, 9 for the amount of stolen bases, and 8 for the RBIs since that is where they are indexed.
'''

df_train = pd.read_csv(train_file, names=COLUMNS, usecols=[8, 9], skipinitialspace=True, skiprows=1)
df_test = pd.read_csv(test_file, names=COLUMNS, usecols=[8, 9], skipinitialspace=True, skiprows=1)


LABEL_COLUMN = "RBI"
df_train[LABEL_COLUMN] = (df_train[LABEL_COLUMN].apply(lambda x: x > 65)).astype(int)
df_test[LABEL_COLUMN] = (df_test[LABEL_COLUMN].apply(lambda x: x > 65)).astype(int)

CONTINUOUS_COLUMNS = ["SB"]

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

sb = tf.contrib.layers.real_valued_column("SB")

#ex. of creating a bucketized_column
#sb_buckets = tf.contrb.layers.bucketized_column(sb, boundaries[0, 5, 10, 15])

model_dir = tempfile.mkdtemp()

# This is where we build our model, train it using the fit method, and then evaluate using evaluate.
model = tf.contrib.learn.LinearClassifier(feature_columns=[sb], model_dir=model_dir, n_classes = 2)

model.fit(input_fn=train_input_fn, steps=200)

results = model.evaluate(input_fn=eval_input_fn, steps=1)
accuracy = sorted(results)[0]
print ("{}: {}".format(accuracy, results[accuracy]))
