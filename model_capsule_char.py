from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import random
random.seed = 42
import pandas as pd
from tensorflow import set_random_seed
set_random_seed(42)
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.layers import *
from classifier_capsule import TextClassifier
from gensim.models.keyedvectors import KeyedVectors
import pickle
import gc


def getClassification(arr):
    arr = list(arr)
    if arr.index(max(arr)) == 0:
        return -2
    elif arr.index(max(arr)) == 1:
        return -1
    elif arr.index(max(arr)) == 2:
        return 0
    else:
        return 1


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = list(map(getClassification, self.model.predict(self.validation_data[0])))
        val_targ = list(map(getClassification, self.validation_data[1]))
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(_val_f1, _val_precision, _val_recall)
        print("max f1")
        print(max(self.val_f1s))
        return


data = pd.read_csv("preprocess/train_char.csv")
data["content"] = data.apply(lambda x: eval(x[1]), axis=1)

validation = pd.read_csv("preprocess/validation_char.csv")
validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)

model_dir = "model_capsule_char/"
maxlen = 1000
max_features = 20000
batch_size = 128
epochs = 15
tokenizer = text.Tokenizer(num_words=None)
tokenizer.fit_on_texts(data["content"].values)
with open('tokenizer_char.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
w2_model = KeyedVectors.load_word2vec_format("word2vec/chars.vector", binary=True, encoding='utf8',
                                             unicode_errors='ignore')
embeddings_index = {}
embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
word2idx = {"_PAD": 0}
vocab_list = [(k, w2_model.wv[k]) for k, v in w2_model.wv.vocab.items()]

for word, i in word_index.items():
    if word in w2_model:
        embedding_vector = w2_model[word]
    else:
        embedding_vector = None
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

X_train = data["content"].values
Y_train_ltc = pd.get_dummies(data["location_traffic_convenience"])[[-2, -1, 0, 1]].values
Y_train_ldfbd = pd.get_dummies(data["location_distance_from_business_district"])[[-2, -1, 0, 1]].values
Y_train_letf = pd.get_dummies(data["location_easy_to_find"])[[-2, -1, 0, 1]].values
Y_train_swt = pd.get_dummies(data["service_wait_time"])[[-2, -1, 0, 1]].values
Y_train_swa = pd.get_dummies(data["service_waiters_attitude"])[[-2, -1, 0, 1]].values
Y_train_spc = pd.get_dummies(data["service_parking_convenience"])[[-2, -1, 0, 1]].values
Y_train_ssp = pd.get_dummies(data["service_serving_speed"])[[-2, -1, 0, 1]].values
Y_train_pl = pd.get_dummies(data["price_level"])[[-2, -1, 0, 1]].values
Y_train_pce = pd.get_dummies(data["price_cost_effective"])[[-2, -1, 0, 1]].values
Y_train_pd = pd.get_dummies(data["price_discount"])[[-2, -1, 0, 1]].values
Y_train_ed = pd.get_dummies(data["environment_decoration"])[[-2, -1, 0, 1]].values
Y_train_en = pd.get_dummies(data["environment_noise"])[[-2, -1, 0, 1]].values
Y_train_es = pd.get_dummies(data["environment_space"])[[-2, -1, 0, 1]].values
Y_train_ec = pd.get_dummies(data["environment_cleaness"])[[-2, -1, 0, 1]].values
Y_train_dp = pd.get_dummies(data["dish_portion"])[[-2, -1, 0, 1]].values
Y_train_dt = pd.get_dummies(data["dish_taste"])[[-2, -1, 0, 1]].values
Y_train_dl = pd.get_dummies(data["dish_look"])[[-2, -1, 0, 1]].values
Y_train_dr = pd.get_dummies(data["dish_recommendation"])[[-2, -1, 0, 1]].values
Y_train_ooe = pd.get_dummies(data["others_overall_experience"])[[-2, -1, 0, 1]].values
Y_train_owta = pd.get_dummies(data["others_willing_to_consume_again"])[[-2, -1, 0, 1]].values

X_validation = validation["content"].values
Y_validation_ltc = pd.get_dummies(validation["location_traffic_convenience"])[[-2, -1, 0, 1]].values
Y_validation_ldfbd = pd.get_dummies(validation["location_distance_from_business_district"])[[-2, -1, 0, 1]].values
Y_validation_letf = pd.get_dummies(validation["location_easy_to_find"])[[-2, -1, 0, 1]].values
Y_validation_swt = pd.get_dummies(validation["service_wait_time"])[[-2, -1, 0, 1]].values
Y_validation_swa = pd.get_dummies(validation["service_waiters_attitude"])[[-2, -1, 0, 1]].values
Y_validation_spc = pd.get_dummies(validation["service_parking_convenience"])[[-2, -1, 0, 1]].values
Y_validation_ssp = pd.get_dummies(validation["service_serving_speed"])[[-2, -1, 0, 1]].values
Y_validation_pl = pd.get_dummies(validation["price_level"])[[-2, -1, 0, 1]].values
Y_validation_pce = pd.get_dummies(validation["price_cost_effective"])[[-2, -1, 0, 1]].values
Y_validation_pd = pd.get_dummies(validation["price_discount"])[[-2, -1, 0, 1]].values
Y_validation_ed = pd.get_dummies(validation["environment_decoration"])[[-2, -1, 0, 1]].values
Y_validation_en = pd.get_dummies(validation["environment_noise"])[[-2, -1, 0, 1]].values
Y_validation_es = pd.get_dummies(validation["environment_space"])[[-2, -1, 0, 1]].values
Y_validation_ec = pd.get_dummies(validation["environment_cleaness"])[[-2, -1, 0, 1]].values
Y_validation_dp = pd.get_dummies(validation["dish_portion"])[[-2, -1, 0, 1]].values
Y_validation_dt = pd.get_dummies(validation["dish_taste"])[[-2, -1, 0, 1]].values
Y_validation_dl = pd.get_dummies(validation["dish_look"])[[-2, -1, 0, 1]].values
Y_validation_dr = pd.get_dummies(validation["dish_recommendation"])[[-2, -1, 0, 1]].values
Y_validation_ooe = pd.get_dummies(validation["others_overall_experience"])[[-2, -1, 0, 1]].values
Y_validation_owta = pd.get_dummies(validation["others_willing_to_consume_again"])[[-2, -1, 0, 1]].values

list_tokenized_train = tokenizer.texts_to_sequences(X_train)
input_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

list_tokenized_validation = tokenizer.texts_to_sequences(X_validation)
input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)

print("model1")
model1 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ltc_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model1.fit(input_train, Y_train_ltc, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_ltc), callbacks=callbacks_list, verbose=2)
del model1
del history
gc.collect()
K.clear_session()

print("model2")
model2 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ldfbd_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model2.fit(input_train, Y_train_ldfbd, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_ldfbd), callbacks=callbacks_list, verbose=2)
del model2
del history
gc.collect()
K.clear_session()

print("model3")
model3 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_letf_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model3.fit(input_train, Y_train_letf, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_letf), callbacks=callbacks_list, verbose=2)
del model3
del history
gc.collect()
K.clear_session()

print("model4")
model4 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_swt_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model4.fit(input_train, Y_train_swt, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_swt), callbacks=callbacks_list, verbose=2)
del model4
del history
gc.collect()
K.clear_session()

print("model5")
model5 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_swa_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model5.fit(input_train, Y_train_swa, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_swa), callbacks=callbacks_list, verbose=2)
del model5
del history
gc.collect()
K.clear_session()

print("model6")
model6 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_spc_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model6.fit(input_train, Y_train_spc, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_spc), callbacks=callbacks_list, verbose=2)
del model6
del history
gc.collect()
K.clear_session()

print("model7")
model7 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ssp_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model7.fit(input_train, Y_train_ssp, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_ssp), callbacks=callbacks_list, verbose=2)
del model7
del history
gc.collect()
K.clear_session()

print("model8")
model8 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_pl_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model8.fit(input_train, Y_train_pl, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_pl), callbacks=callbacks_list, verbose=2)
del model8
del history
gc.collect()
K.clear_session()

print("model9")
model9 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_pce_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model9.fit(input_train, Y_train_pce, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation_pce), callbacks=callbacks_list, verbose=2)
del model9
del history
gc.collect()
K.clear_session()

print("model10")
model10 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_pd_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model10.fit(input_train, Y_train_pd, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_pd), callbacks=callbacks_list, verbose=2)
del model10
del history
gc.collect()
K.clear_session()

print("model11")
model11 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ed_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model11.fit(input_train, Y_train_ed, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_ed), callbacks=callbacks_list, verbose=2)
del model11
del history
gc.collect()
K.clear_session()

print("model12")
model12 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_en_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]

history = model12.fit(input_train, Y_train_en, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_en), callbacks=callbacks_list, verbose=2)
del model12
del history
gc.collect()
K.clear_session()

print("model13")
model13 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_es_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]

history = model13.fit(input_train, Y_train_es, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_es), callbacks=callbacks_list, verbose=2)
del model13
del history
gc.collect()
K.clear_session()

print("model14")
model14 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ec_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]

history = model14.fit(input_train, Y_train_ec, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_ec), callbacks=callbacks_list, verbose=2)
del model14
del history
gc.collect()
K.clear_session()

print("model15")
model15 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_dp_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model15.fit(input_train, Y_train_dp, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_dp), callbacks=callbacks_list, verbose=2)
del model15
del history
gc.collect()
K.clear_session()

print("model16")
model16 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_dt_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model16.fit(input_train, Y_train_dt, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_dt), callbacks=callbacks_list, verbose=2)
del model16
del history
gc.collect()
K.clear_session()

print("model17")
model17 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_dl_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model17.fit(input_train, Y_train_dl, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_dl), callbacks=callbacks_list, verbose=2)
del model17
del history
gc.collect()
K.clear_session()

print("model18")
model18 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_dr_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model18.fit(input_train, Y_train_dr, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_dr), callbacks=callbacks_list, verbose=2)
del model18
del history
gc.collect()
K.clear_session()

print("model19")
model19 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_ooe_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model19.fit(input_train, Y_train_ooe, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_ooe), callbacks=callbacks_list, verbose=2)
del model19
del history
gc.collect()
K.clear_session()

print("model20")
model20 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
file_path = model_dir + "model_owta_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=2, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]

history = model20.fit(input_train, Y_train_owta, batch_size=batch_size, epochs=epochs,
                      validation_data=(input_validation, Y_validation_owta), callbacks=callbacks_list, verbose=2)
del model20
del history
gc.collect()
K.clear_session()


