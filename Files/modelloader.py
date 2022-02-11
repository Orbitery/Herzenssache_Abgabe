from keras.models import load_model
import joblib

def modelload(is_binary_classifier,model_name): #Load the pretrained Model 
    if(is_binary_classifier==True):
        if (model_name == "CNN"):
            model = load_model("./CNN_bin/model_bin.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_bin/model_bin.hdf5")
        elif(model_name=="Resnet"):
            model = load_model("./Resnet_bin/model_bin.hdf5")            
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_bin/random_forest.joblib")
    elif(is_binary_classifier ==False):
        if (model_name == "CNN"):
            model = load_model("./CNN_multi/model_multi.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_multi/model_multi.hdf5")
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_multi/random_forest.joblib")
        elif(model_name=="Resnet"):
            model = load_model("./Resnet_multi/model_multi.hdf5")     

    print ("Model loaded")
    return (model)