from re import M
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics

def CNN_Model(X_train, y_train, X_test, y_test):
    print("CNN Model was chosen")
    #Definieren der CNN Architektur. Hierbei wurde sich bei der Architektur an dem Paper "ECG Heartbeat Classification Using Convolutional Neural Networks" von Xu und Liu, 2020 orientiert. 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = models.Sequential()
    model.add(layers.GaussianNoise(0.1))
    model.add(layers.Conv1D(64, 5, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=3, batch_size=1024,validation_data=(X_test, y_test), callbacks=[callback])

    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    # list all data in 
    print(history.history)
    print(history.history.keys())
    return (model, history)

def LSTM(X_train, y_train, X_test, y_test):
    print("LSTM Model was chosen")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = models.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, stateful=False, input_shape = X_train[0].shape))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=2, batch_size=1024,validation_data=(X_test, y_test), callbacks=[callback])
    model.build()
    model.summary()
    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    # list all data in 
    print(history.history)
    print(history.history.keys())
    return (model, history)

def RandomForrest(X_train, y_train, X_test, y_test):
    print("Random Forrest Model was chosen")
    X_train = X_train[:,:,0]
    X_test = X_test[:,:,0]

    #cross validation
    m = RandomForestClassifier(n_jobs=-1)
    history = m.fit(X_train, y_train)
    loss = metrics.log_loss(y_test,m.predict_proba(X_test))
    print("Loss is {}".format(loss))
    accuracy = metrics.accuracy_score(y_test,m.predict(X_test))
    print ("Acc is {}".format(accuracy))
    return (m, history)

def Resnet(X_train, y_train, X_test, y_test):
    print("Resnet Model was chosen")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    def get_resnet_model(categories=2):
        def residual_block(X, kernels, stride):
            out = tf.keras.layers.Conv1D(kernels, stride, padding='same')(X)
            out = tf.keras.layers.ReLU()(out)
            out = tf.keras.layers.Conv1D(kernels, stride, padding='same')(out)
            out = tf.keras.layers.add([X, out])
            out = tf.keras.layers.ReLU()(out)
            out = tf.keras.layers.MaxPool1D(5, 2)(out)
            return out

        kernels = 32
        stride = 5

        inputs = tf.keras.layers.Input(shape=(X_train.shape[1],1))
        X = tf.keras.layers.Conv1D(kernels, stride)(inputs)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = residual_block(X, kernels, stride)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(32, activation='relu')(X)
        X = tf.keras.layers.Dense(32, activation='relu')(X)
        output = tf.keras.layers.Dense(y_train.shape[1], activation='softmax')(X)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    model = get_resnet_model()
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    history = model.fit(X_train, y_train, epochs=3,validation_data=(X_test, y_test), batch_size=1028, callbacks=[callback])
    model.build(input_shape=(X_train.shape[1],1))
    model.summary()
    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    return (model, history)
