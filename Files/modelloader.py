from keras.models import load_model
import joblib

def modelload(is_binary_classifier,model_name): #Load the pretrained Model 
    if(is_binary_classifier=="True"):
        if (model_name == "CNN"):
            model = load_model("./CNN_Model/model_bin.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_Model/model_bin.hdf5")
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_bin/random_forest.joblib")
    elif(is_binary_classifier =="False"):
        if (model_name == "CNN"):
            model = load_model("./CNN_Model/model_multi.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_Model/model_multi.hdf5")
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_multi/random_forest.joblib")
    print ("Model loaded")
    return (model)