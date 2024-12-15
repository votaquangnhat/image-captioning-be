import os
import argparse
import numpy as np
import tensorflow as tf
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model

with open(os.path.join(r"model/CoCo_transform_train2017.json"), 'r') as file:
    mapping_train_origin = json.load(file)
mapping_train = dict()

for key in mapping_train_origin:
    mapping_train[key.split('.')[0]] = mapping_train_origin[key]

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'seqstart ' + " ".join([word for word in caption.split() if len(word)>1]) + ' seqend'
            captions[i] = caption

clean(mapping_train)

all_train_captions = []
for key in mapping_train:
    for caption in mapping_train[key]:
        all_train_captions.append(caption)


class Image_to_Text():
    def __init__(self, image: Image, image_path = None, model_path = r"model/VGG&LSTM_CoCo_model.keras", all_train_captions = all_train_captions):
        image_id = None
        if not image_path and isinstance(image_path, str):    
            image_name_file = image_path.split('/')[-1]
            image_id = image_name_file.split('.')[0]
        self.image_id = image_id
        self.image_path = image_path
        self.model_path = model_path
        self.all_train_captions = all_train_captions
        self.image = image
    
    def extract_features(self):
        # Load model VGG16 to extract features from image
        #load vgg16 model
        model = VGG16()
        # restructure the model
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

        # extract feature from image
        features = {}

        image = None
        if not self.image_path and isinstance(self.image_path, str):
            image = load_img(self.image_path, target_size=(224, 224))
        else:
            image = self.image.resize((224,224))
        # conver image pixel to np array
        image = img_to_array(image)
        # reshape data for model 
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for vgg
        image = preprocess_input(image)
        # extract features
        feature = model.predict(image, verbose=0)
        # get image i d
        # image_name_file = self.image_path.split('/')[-1]
        image_id = self.image_id#image_name_file.split('.')[0]
        # store feature 
        features[image_id] = feature

        return features
    
    def LSTM_structure_load(self):
        model = load_model(self.model_path)
        return model
    
    def to_Token(self):
        # tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.all_train_captions)
        vocab_size = len(tokenizer.word_index) + 1

        max_length = max(len(caption.split()) for caption in all_train_captions)
        max_length

        return tokenizer, vocab_size, max_length

    def idx_to_word(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
        
    def predict_caption(self, model, image, tokenizer, max_length):
        # add start tag for generation process
        in_text = 'seqstart'
        # iterate over the max length of sequence
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length, padding='post')
            # predict next word
            yhat = model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = self.idx_to_word(yhat, tokenizer)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'seqend':
                break
        return in_text
    
    def to_text(self):
        features = self.extract_features()
        LSTM_model = self.LSTM_structure_load()
        tokenizer, vocab_size, max_length = self.to_Token()

        y_pred = self.predict_caption(LSTM_model, features[self.image_id], tokenizer, max_length)
        sequences = y_pred.split()
        sequences = sequences[1:len(sequences) - 1]
        y_pred = " ".join(sequences)
        return y_pred
    
    def visualize_the_result(self):
    # Load the image

        image_path = self.image_path
        image = mpimg.imread(image_path)
        plt.imshow(image)
        plt.axis('off')  # Tắt trục để hiển thị rõ hơn
        # Predicted caption
        y_pred = self.to_text()
        print( "CAPTION: ", y_pred)
        plt.title(y_pred)
        plt.show()

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Generate captions for an image using a pre-trained VGG & LSTM model.")
    # parser.add_argument("image_path", type=str, help="Path to the input image")

    # args = parser.parse_args()

    # image_path = rf"{args.image_path}"
    image_path = r"test1.jpg"
    model_path = r"model/VGG&LSTM_CoCo_model.keras"

    text_generator = Image_to_Text(Image.open(image_path).convert('RGB'))
    print(text_generator.to_text())