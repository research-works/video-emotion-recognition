import os
import numpy as np
import librosa
from PIL import Image
import utils.audio_utils as audio_utils
import tensorflow.keras as keras

class FaceDataGenerator(keras.utils.Sequence):

    def __init__(self, file_names, labels, batch_size, image_width, image_height):
        self.file_names = file_names
        self.batch_size = batch_size
        self.labels = labels #emotions
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return np.ceil(len(self.file_names) / float(self.batch_size)).astype(np.int)


    def __getitem__(self, index):
        data_x = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        for file_name in self.file_names[start:end]:
            image = Image.open(file_name)
            image.load()
            image = image.resize((self.image_width, self.image_height))
            data_x.append(np.array(image))
        data_x = np.array(data_x)
        data_y = self.labels[start:end]
        return data_x, data_y


class AudioDataGenerator(keras.utils.Sequence):

    def __init__(self, file_names, labels, batch_size, image_width, image_height):
        self.file_names = file_names
        self.batch_size = batch_size
        self.labels = labels #emotions
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return np.ceil(len(self.file_names) / float(self.batch_size)).astype(np.int)


    def __getitem__(self, index):
        data_x = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        for file_name in self.file_names[start:end]:
            spect = np.load(file_name, allow_pickle = True)
            spect_static = spect[:,:,0]
            spect_delta = spect[:,:,1]
            spect_delta2 = spect[:,:,2]

            spect_static = audio_utils.resize(spect_static, 
                                            self.image_width, 
                                            self.image_height)
            spect_delta = audio_utils.resize(spect_delta, 
                                            self.image_width, 
                                            self.image_height)
            spect_delta2 = audio_utils.resize(spect_delta2, 
                                            self.image_width, 
                                            self.image_height)

            spect = np.zeros((self.image_width, self.image_height, 3))
            spect[:,:,0] = spect_static
            spect[:,:,1] = spect_delta
            spect[:,:,2] = spect_delta2

            data_x.append(spect)
        data_x = np.array(data_x)
        data_y = self.labels[start:end]
        return data_x, data_y


class MultimodalDataGenerator(keras.utils.Sequence):
    def __init__(self, file_names_face, file_names_audio, labels, batch_size, image_width, image_height):
        self.face_gen = FaceDataGenerator(file_names_face, 
                                            labels, 
                                            batch_size, 
                                            image_width, 
                                            image_height)
        self.audio_gen = AudioDataGenerator(file_names_audio, 
                                                labels, 
                                                batch_size, 
                                                image_width, 
                                                image_height)

    def __len__(self):
        return self.face_gen.__len__()

    def __getitem__(self, index):
        data_x_face, data_y = self.face_gen.__getitem__(index)
        data_x_audio, data_y = self.audio_gen.__getitem__(index)
        return [data_x_face, data_x_audio], data_y
