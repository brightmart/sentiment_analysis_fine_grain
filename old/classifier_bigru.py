import keras
from keras import Model
from keras.layers import *
from JoinAttLayer import Attention


class TextClassifier():

    def model(self, embeddings_matrix, maxlen, word_index, num_class):
        inp = Input(shape=(maxlen,))
        encode = Bidirectional(CuDNNGRU(128, return_sequences=True))
        encode2 = Bidirectional(CuDNNGRU(128, return_sequences=True))
        attention = Attention(maxlen)
        x_4 = Embedding(len(word_index) + 1,
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        trainable=True)(inp)
        x_3 = SpatialDropout1D(0.2)(x_4)
        x_3 = encode(x_3)
        x_3 = Dropout(0.2)(x_3)
        x_3 = encode2(x_3)
        x_3 = Dropout(0.2)(x_3)
        avg_pool_3 = GlobalAveragePooling1D()(x_3)
        max_pool_3 = GlobalMaxPooling1D()(x_3)
        attention_3 = attention(x_3)
        x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3], name="fc")
        x = Dense(num_class, activation="sigmoid")(x)

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)
        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        model = Model(inputs=inp, outputs=x)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam)
        return model
