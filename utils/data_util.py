import os
import numpy as np
import librosa
from PIL import Image
import utils.audio_utils as audio_utils
import utils.video_utils as video_utils
import tensorflow.keras as keras

BASE_DIR = '/home/reckoner1429/Documents/projects/video-emotion-recognition/data/RAVDESS'
PREPROCESSED_VIDEO_DIR = 'Video_preprocessing_output'
PREPROCESSED_AUDIO_DIR = 'Audio_preprocessing_output'
EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad','angry','fearful','disgust','surprised']

SAMPLE_RATE = 44100
DURATION = 2.5
OFFSET = 0.8

OUTPUT_IMAGE_WIDTH = 299
OUTPUT_IMAGE_HEIGHT = 299
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
# file1 - calm - [0,1,0,0,0,0,0,0]
# fil2 - angry - [0,0,0,0,1,0,0,0]
def one_hot(i, n=8):
    arr = np.zeros(8)
    arr[i] = 1
    return arr

def zero_pad(x, n):
    diff = n - x.shape[0]
    before = diff // 2
    after = diff - before
    x = np.pad(x, (before, after))
    return x

def preprocess_facial_data():
    DATASET_DIR = BASE_DIR + '/' + 'ravdess_speech_videos'
    OUTPUT_DIR = BASE_DIR + '/' + PREPROCESSED_VIDEO_DIR
    for actor_folder in os.listdir(DATASET_DIR):
        print(actor_folder)
        act_no = actor_folder.split('_')[1]
        print(act_no)
        for mp4file in os.listdir(DATASET_DIR + "/" + actor_folder):

            if mp4file.endswith(".mp4") and mp4file.startswith('01'):
                print(mp4file)
                
                input_video_path = DATASET_DIR + '/' + actor_folder + '/' + mp4file
                output_dir_path = OUTPUT_DIR + '/' + actor_folder
                video_utils.preprocess_video(input_video_path, output_dir_path, 10)

def load_facial_data(preprocess = False):
    if(preprocess == True):
        preprocess_facial_data()

    X, Y = [], []
    base_path = BASE_DIR + '/' + PREPROCESSED_VIDEO_DIR
    for actor_folder in os.listdir(base_path):
        actor_path = base_path + '/' + actor_folder + '/' + 'subtracted_frames'
        files_list = []
        for image_file in os.listdir(actor_path):
            files_list.append(image_file)
        files_list.sort()
        for image_file in files_list:
            image_path = actor_path + '/' + image_file
            image = Image.open(image_path)
            image.load()
            image = image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)) # 48 * 48
            image = np.asarray(image, dtype = 'float32')
            # image = image / np.linalg.norm(image)
            # print(image.shape)
            # image = image.reshape(image.shape[0], image.shape[1], 1) # 48,48,1
            X.append(image)

            em_id = int(image_file.split('-')[2])
            one_hot_em = one_hot(em_id - 1)
            # print(one_hot_em)
            Y.append(one_hot_em)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_facial_filenames():
    X, Y = [], []
    base_path = BASE_DIR + '/' + PREPROCESSED_VIDEO_DIR
    for actor_folder in os.listdir(base_path):
        actor_path = base_path + '/' + actor_folder + '/' + 'subtracted_frames'
        files_list = []
        for image_file in os.listdir(actor_path):
            files_list.append(image_file)
        files_list.sort()
        for image_file in files_list:
            image_path = actor_path + '/' + image_file
            # image = Image.open(image_path)
            # image.load()
            # image = image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)) # 48 * 48
            # image = np.asarray(image, dtype = 'float32')
            # # image = image / np.linalg.norm(image)
            # # print(image.shape)
            # # image = image.reshape(image.shape[0], image.shape[1], 1) # 48,48,1
            X.append(image_path)

            em_id = int(image_file.split('-')[2])
            one_hot_em = one_hot(em_id - 1)
            # print(one_hot_em)
            Y.append(one_hot_em)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def preprocess_audio_data():
    DATASET_DIR = BASE_DIR + '/' + 'Audio_Speech_Actors_01-24'
    OUTPUT_DIR = BASE_DIR + '/' + PREPROCESSED_AUDIO_DIR
    for actor_folder in os.listdir(DATASET_DIR):
        print(actor_folder)
        act_no = actor_folder.split('_')[1]
        print(act_no)
        for wavfile in os.listdir(DATASET_DIR + "/" + actor_folder):
                
            if wavfile.endswith(".wav"):
                print(wavfile)
                
                input_video_path = DATASET_DIR + '/' + actor_folder + '/' + wavfile
                output_dir_path = OUTPUT_DIR + '/' + actor_folder
                audio_utils.preprocess_audio(input_video_path, output_dir_path)

def load_audio_data(preprocess = False):
    if(preprocess == True):
        preprocess_audio_data()

    X, Y = [], []
    base_path = BASE_DIR + '/' + PREPROCESSED_AUDIO_DIR
    for actor_folder in os.listdir(base_path):
        actor_path = base_path + '/' + actor_folder
        files_list = []
        for audio_file in os.listdir(actor_path):
            files_list.append(audio_file)
        files_list.sort()
        for audio_file in files_list:
            audio_path = actor_path + '/' + audio_file
            S_input = np.load(audio_path)
            # print(S_input.shape)
            X.append(S_input) # (216,1)

            em_id = int(audio_file.split('-')[2])
            one_hot_em = one_hot(em_id - 1)
            # print(one_hot_em.shape)
            Y.append(one_hot_em)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_audio_filenames():
    X, Y = [], []
    base_path = BASE_DIR + '/' + PREPROCESSED_AUDIO_DIR
    for actor_folder in os.listdir(base_path):
        actor_path = base_path + '/' + actor_folder
        files_list = []
        for audio_file in os.listdir(actor_path):
            files_list.append(audio_file)
        files_list.sort()
        for audio_file in files_list:
            audio_path = actor_path + '/' + audio_file
            # S_input = np.load(audio_path)
            # # print(S_input.shape)
            # X.append(S_input) # (216,1)
            X.append(audio_path)

            em_id = int(audio_file.split('-')[2])
            one_hot_em = one_hot(em_id - 1)
            # print(one_hot_em.shape)
            Y.append(one_hot_em)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_names, labels, batch_size):
        # if(batch_size > buffer_size):
        #     raise ValueError('buffer_size must be greater than or equal to batch_size')
        self.file_names = file_names
        self.batch_size = batch_size
        # self.buffer = []
        # self.buffer_size = buffer_size
        self.labels = labels

    def __len__(self):
        return np.ceil(len(self.file_names) /float(self.batch_size)).astype(np.int)

    # def get_data(self, index):
    #     if(len(self.buffer) == 0):
    #         for i in range(index, min(index + self.buffer_size, len(self.file_names))):
    #             image = np.asarray(Image.open(self.file_names[i]).load())
    #             self.buffer.append(image)

    #     start = index
    #     end = min(index + self.batch_size, len(self.file_names))
    #     data = np.array(self.buffer[start:end])
    #     del self.buffer[start:end]
    #     return data, self.labels[start:end]

    def __getitem__(self, index):
        data_x = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        for file_name in self.file_names[start:end]:
            image = Image.open(file_name)
            image.load()
            image = image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
            data_x.append(np.array(image))
        data_x = np.array(data_x)
        data_y = self.labels[start:end]

        return [data_x, data_x], data_y

# X, Y = load_facial_filenames()
# print(len(X))
# datagen = DataGenerator(X, Y, 8)
# for i in range(180):
#     # print(i)
#     X, Y = datagen.__getitem__(i)
#     print(X.shape)

# print(datagen.__len__())