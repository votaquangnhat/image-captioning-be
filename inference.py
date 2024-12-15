from dumbModel.dumb_model_infer import infer
import Infer.transformer_model_infer as trans
import Infer.transformer_model_pre_infer as transpre
import Infer.cnn_lstm_infer as cnn_lstm

def transformer_inference(image):
    caption = trans.main_(image)
    return caption

def transformerpre_inference(image):
    caption = transpre.main_(image)
    return caption

def cnn_lstm_inference(image):
    text_generator = cnn_lstm.Image_to_Text(image)
    caption = text_generator.to_text()
    return caption

def generateCaption(image, model = "dumb"):
    if model == "dumb":
        return infer(image)
    elif model == "trans":
        return transformer_inference(image)
    elif model == "transpre":
        return transformerpre_inference(image)
    elif model == "cnnlstm":
        return cnn_lstm_inference(image)
    
    return 'Cannot use this model now'