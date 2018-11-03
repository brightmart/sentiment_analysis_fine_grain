from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import gc
import pandas as pd
import pickle
import numpy as np
np.random.seed(16)
from tensorflow import set_random_seed
set_random_seed(16)
from keras.layers import *
from keras.preprocessing import sequence
from gensim.models.keyedvectors import KeyedVectors
from classifier_rcnn import TextClassifier


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


if __name__ == "__main__":
    with open('tokenizer_char.pickle', 'rb') as handle:
        maxlen = 1000
        model_dir = "model_rcnn_char/"
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        validation = pd.read_csv("preprocess/validation_char.csv")
        validation["content"] = validation.apply(lambda x: eval(x[1]), axis=1)
        X_test = validation["content"].values
        list_tokenized_validation = tokenizer.texts_to_sequences(X_test)
        input_validation = sequence.pad_sequences(list_tokenized_validation, maxlen=maxlen)
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

        submit = pd.read_csv("ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
        submit_prob = pd.read_csv("ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")

        model1 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model1.load_weights(model_dir + "model_ltc_02.hdf5")
        submit["location_traffic_convenience"] = list(map(getClassification, model1.predict(input_validation)))
        submit_prob["location_traffic_convenience"] = list(model1.predict(input_validation))
        del model1
        gc.collect()
        K.clear_session()

        model2 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model2.load_weights(model_dir + "model_ldfbd_02.hdf5")
        submit["location_distance_from_business_district"] = list(
            map(getClassification, model2.predict(input_validation)))
        submit_prob["location_distance_from_business_district"] = list(model2.predict(input_validation))
        del model2
        gc.collect()
        K.clear_session()

        model3 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model3.load_weights(model_dir + "model_letf_02.hdf5")
        submit["location_easy_to_find"] = list(map(getClassification, model3.predict(input_validation)))
        submit_prob["location_easy_to_find"] = list(model3.predict(input_validation))
        del model3
        gc.collect()
        K.clear_session()

        model4 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model4.load_weights(model_dir + "model_swt_02.hdf5")
        submit["service_wait_time"] = list(map(getClassification, model4.predict(input_validation)))
        submit_prob["service_wait_time"] = list(model4.predict(input_validation))
        del model4
        gc.collect()
        K.clear_session()

        model5 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model5.load_weights(model_dir + "model_swa_02.hdf5")
        submit["service_waiters_attitude"] = list(map(getClassification, model5.predict(input_validation)))
        submit_prob["service_waiters_attitude"] = list(model5.predict(input_validation))
        del model5
        gc.collect()
        K.clear_session()

        model6 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model6.load_weights(model_dir + "model_spc_01.hdf5")
        submit["service_parking_convenience"] = list(map(getClassification, model6.predict(input_validation)))
        submit_prob["service_parking_convenience"] = list(model6.predict(input_validation))
        del model6
        gc.collect()
        K.clear_session()

        model7 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model7.load_weights(model_dir + "model_ssp_02.hdf5")
        submit["service_serving_speed"] = list(map(getClassification, model7.predict(input_validation)))
        submit_prob["service_serving_speed"] = list(model7.predict(input_validation))
        del model7
        gc.collect()
        K.clear_session()

        model8 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model8.load_weights(model_dir + "model_pl_02.hdf5")
        submit["price_level"] = list(map(getClassification, model8.predict(input_validation)))
        submit_prob["price_level"] = list(model8.predict(input_validation))
        del model8
        gc.collect()
        K.clear_session()

        model9 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model9.load_weights(model_dir + "model_pce_02.hdf5")
        submit["price_cost_effective"] = list(map(getClassification, model9.predict(input_validation)))
        submit_prob["price_cost_effective"] = list(model9.predict(input_validation))
        del model9
        gc.collect()
        K.clear_session()

        model10 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model10.load_weights(model_dir + "model_pd_02.hdf5")
        submit["price_discount"] = list(map(getClassification, model10.predict(input_validation)))
        submit_prob["price_discount"] = list(model10.predict(input_validation))
        del model10
        gc.collect()
        K.clear_session()

        model11 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model11.load_weights(model_dir + "model_ed_02.hdf5")
        submit["environment_decoration"] = list(map(getClassification, model11.predict(input_validation)))
        submit_prob["environment_decoration"] = list(model11.predict(input_validation))
        del model11
        gc.collect()
        K.clear_session()

        model12 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model12.load_weights(model_dir + "model_en_02.hdf5")
        submit["environment_noise"] = list(map(getClassification, model12.predict(input_validation)))
        submit_prob["environment_noise"] = list(model12.predict(input_validation))
        del model12
        gc.collect()
        K.clear_session()

        model13 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model13.load_weights(model_dir + "model_es_01.hdf5")
        submit["environment_space"] = list(map(getClassification, model13.predict(input_validation)))
        submit_prob["environment_space"] = list(model13.predict(input_validation))
        del model13
        gc.collect()
        K.clear_session()

        model14 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model14.load_weights(model_dir + "model_ec_02.hdf5")
        submit["environment_cleaness"] = list(map(getClassification, model14.predict(input_validation)))
        submit_prob["environment_cleaness"] = list(model14.predict(input_validation))
        del model14
        gc.collect()
        K.clear_session()

        model15 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model15.load_weights(model_dir + "model_dp_02.hdf5")
        submit["dish_portion"] = list(map(getClassification, model15.predict(input_validation)))
        submit_prob["dish_portion"] = list(model15.predict(input_validation))
        del model15
        gc.collect()
        K.clear_session()

        model16 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model16.load_weights(model_dir + "model_dt_02.hdf5")
        submit["dish_taste"] = list(map(getClassification, model16.predict(input_validation)))
        submit_prob["dish_taste"] = list(model16.predict(input_validation))
        del model16
        gc.collect()
        K.clear_session()

        model17 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model17.load_weights(model_dir + "model_dl_02.hdf5")
        submit["dish_look"] = list(map(getClassification, model17.predict(input_validation)))
        submit_prob["dish_look"] = list(model17.predict(input_validation))
        del model17
        gc.collect()
        K.clear_session()

        model18 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model18.load_weights(model_dir + "model_dr_02.hdf5")
        submit["dish_recommendation"] = list(map(getClassification, model18.predict(input_validation)))
        submit_prob["dish_recommendation"] = list(model18.predict(input_validation))
        del model18
        gc.collect()
        K.clear_session()

        model19 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model19.load_weights(model_dir + "model_ooe_02.hdf5")
        submit["others_overall_experience"] = list(map(getClassification, model19.predict(input_validation)))
        submit_prob["others_overall_experience"] = list(model19.predict(input_validation))
        del model19
        gc.collect()
        K.clear_session()

        model20 = TextClassifier().model(embeddings_matrix, maxlen, word_index, 4)
        model20.load_weights(model_dir + "model_owta_02.hdf5")
        submit["others_willing_to_consume_again"] = list(map(getClassification, model20.predict(input_validation)))
        submit_prob["others_willing_to_consume_again"] = list(model20.predict(input_validation))
        del model20
        gc.collect()
        K.clear_session()

        submit.to_csv("validation_rcnn_char.csv", index=None)
        submit_prob.to_csv("validation_rcnn_char_prob.csv", index=None)