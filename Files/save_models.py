import os
import joblib
def Saver(model,bin,modelname):
    if (modelname =="CNN"):
        if not os.path.exists("./CNN_bin/"):
            os.mkdir("./CNN_bin/")
        else:
            pass
        if not os.path.exists("./CNN_multi/"):
            os.mkdir("./CNN_multi/")
        else:
            pass
        try:    
            with open('./CNN_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin==True):
            if os.path.exists("./CNN_bin/model_bin.hdf5"):
                os.remove("./CNN_bin/model_bin.hdf5")
            else:
                pass
            model.save("./CNN_bin/model_bin.hdf5")

        elif (bin==False):
            if os.path.exists("./CNN_multi/model_multi.hdf5"):
                os.remove("./CNN_multi/model_multi.hdf5")
            else:
                pass
            model.save("./CNN_multi/model_multi.hdf5")
    elif(modelname=="LSTM"):
        if not os.path.exists("./LSTM_bin/"):
            os.mkdir("./LSTM_bin/")
        else:
            pass
        if not os.path.exists("./LSTM_multi/"):
            os.mkdir("./LSTM_multi/")
        else:
            pass
        try:    
            with open('./LSTM_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin==True):
            if os.path.exists("./LSTM_bin/model_bin.hdf5"):
                os.remove("./LSTM_bin/model_bin.hdf5")
            else:
                pass
            model.save("./LSTM_bin/model_bin.hdf5")

        elif (bin==False):
            if os.path.exists("./LSTM_multi/model_multi.hdf5"):
                os.remove("./LSTM_multi/model_multi.hdf5")
            else:
                pass
            model.save("./LSTM_multi/model_multi.hdf5")
    elif(modelname=="RandomForrest"):
        if not os.path.exists("./RF_bin/"):
            os.mkdir("./RF_bin/")
        else:
            pass    
        if not os.path.exists("./RF_multi/"):
            os.mkdir("./RF_multi/")
        else:
            pass
        try:
            with open('./RF_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin==True):
            if os.path.exists("./RF_bin/random_forest.joblib"):
                os.remove("./RF_bin/random_forest.joblib")
            else:
                pass
            joblib.dump(model, "./RF_bin/random_forest.joblib")

        elif (bin==False):
            if os.path.exists("./RF_multi/random_forest.joblib"):
                os.remove("./RF_multi/random_forest.joblib")
            else:
                pass
            joblib.dump(model, "./RF_bin/random_forest.joblib")