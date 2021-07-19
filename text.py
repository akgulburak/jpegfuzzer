import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

from tensorflow.keras.callbacks import ModelCheckpoint
import re
#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

def read_bytes(file):
	text = []
	count=0
	while True:
		if count<6:
			count+=1
			continue
		byte = file.read(1)
		if not byte or count>9196:
			break
		byte = int.from_bytes(byte,"big")
		text.append(byte)
	return text

names = os.listdir("dataset")
names.sort()
files = names[:100]
dataset = []
x_files = []
raw = []
vocab = set()

for filename in files:
	with open("dataset"+"/"+filename, 'r') as file:
		file = open("dataset/"+filename,"rb")
		raw.append(file.read())
		file.close()

		file = open("dataset/"+filename,"rb")
		text = read_bytes(file)		
		file.close()

		file = open("tmp.jpg","rb")
		x_file = read_bytes(file)		
		file.close()

		x_files.append(x_file)
		dataset.append(text)

		tmp = sorted(set(text))
		vocab = vocab.union(tmp)

starting = raw[0][:6]
ending = raw[0][-2:]
#print(text[:250])
#chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
#def text_from_ids(ids):
#  return tf.strings.reduce_join(ids, axis=-1)
#
#for char in chars[:10]:
#    print(char.numpy().decode('utf-8'))

seq_length = 9188
examples_per_epoch = 1

# Batch size
#BATCH_SIZE = 64
#
## Buffer size to shuffle the dataset
## (TF data is designed to work with possibly infinite sequences,
## so it doesn't attempt to shuffle the entire sequence in memory. Instead,
## it maintains a buffer in which it shuffles elements).
#BUFFER_SIZE = 10000
#
#dataset = (
#    dataset
#    .shuffle(BUFFER_SIZE)
#    .batch(BATCH_SIZE, drop_remainder=True)
#    .prefetch(tf.data.experimental.AUTOTUNE))

#print(text_from_ids(chars))

#print(chars)
#text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss,metrics=["accuracy"])

dataset = np.array(dataset)
x_files = np.array(x_files)
# Directory where the checkpoints will be saved
checkpoint_dir = './checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

filepath = "checkpoints/epoch{epoch:02d}.ckpt"
checkpoint = ModelCheckpoint(filepath=filepath, 
                             verbose=1)

EPOCHS = 10
def train():
	model.fit(x_files,dataset,batch_size=8,epochs=EPOCHS,callbacks=[checkpoint])

def predictions(data,path):
	prediction = model.predict(data)

	sampled_indices = tf.random.categorical(prediction[:,0,:], num_samples=1).numpy()
	tmp = []
	for i in range(len(sampled_indices)):
		without_null = sampled_indices[i][0].tostring().rstrip(b'\x00')
		if without_null == b'': 
			tmp.append(b'\x00')
		else:
			tmp.append(without_null)
	#print(sampled_indices[0][0].tostring())
	#sampled_indices = bytes(sampled_indices)
	sampled_indices = tmp

	sampled_indices = [starting] + sampled_indices + [ending]
	sampled_indices = b''.join(sampled_indices)
	file = open(path+"prediction.jpg","wb")
	file.write(sampled_indices)
	file.close()

train()

### Prediction part
file_names = os.listdir("testing")
files = []
for file in file_names:
	files.append("testing/"+file)

for i in range(len(files)):
	print(i)
	file = open(files[i],"rb")
	test_data = read_bytes(file)
	file.close()

	checkpoint_path = "checkpoints/epoch10.ckpt"
	model.load_weights(checkpoint_path)
	path = "predicts/"+str(i)+"_"
	predictions(test_data,path)
###

#for data in dataset[:1]:
#	example_batch_predictions = model(np.array([data]))
#	#print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
#	sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#	sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#
#	example_batch_loss = loss(data, example_batch_predictions)
#	mean_loss = example_batch_loss.numpy().mean()
#	print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
#	print("Mean loss:        ", mean_loss)
#
#	#print(sampled_indices)
