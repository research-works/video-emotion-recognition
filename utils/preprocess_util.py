import os
import numpy as np
import utils.config as config
import utils.audio_utils as audio_utils
import utils.video_utils as video_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from abc import ABC, abstractmethod

def zero_pad(x, n):
    diff = n - x.shape[0]
    before = diff // 2
    after = diff - before
    x = np.pad(x, (before, after))
    return x


class Preprocess(ABC):
    @abstractmethod
    def process_audio(self):
        pass

    @abstractmethod
    def process_video(self):
        pass

    @abstractmethod
    def load_audio_filenames(self):
        pass

    @abstractmethod
    def load_visual_filenames(self):
        pass
    
    @property
    @abstractmethod
    def emotion_classes():
        pass

    @property
    @abstractmethod
    def audio_duration():
        pass

    @property
    @abstractmethod
    def sample_rate():
        pass
    
    @property
    @abstractmethod
    def offset():
        pass

class RAVDESS(Preprocess):
    EMOTION_CLASSES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    DURATION = 2.5
    OFFSET = 0.8
    SAMPLE_RATE = 441000

    def __init__(self):
        self.dataset_config = config.DatasetConfig("RAVDESS")
        self.DATASET_BASE_DIR = self.dataset_config.DATASET_BASE_DIR
        self.PREPROCESSED_AUDIO_SAVE_DIR = self.dataset_config.PREPROCESSED_AUDIO_SAVE_DIR
        self.PREPROCESSED_AUDIO_DIR = self.dataset_config.PREPROCESSED_AUDIO_DIR
        self.PREPROCESSED_VIDEO_SAVE_DIR = self.dataset_config.PREPROCESSED_VIDEO_SAVE_DIR
        self.PREPROCESSED_VIDEO_DIR = self.dataset_config.PREPROCESSED_VIDEO_DIR
        self.MODEL_SAVE_DIR = self.dataset_config.MODEL_SAVE_DIR

    def process_audio(self):
        DATASET_DIR = self.DATASET_BASE_DIR + '/' + 'Audio_Speech_Actors_01-24'
        OUTPUT_DIR = self.PREPROCESSED_AUDIO_SAVE_DIR
        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder.split('_')[1]
            print(act_no)
            for wavfile in os.listdir(DATASET_DIR + "/" + actor_folder):

                if wavfile.endswith(".wav"):
                    print(wavfile)

                    input_video_path = DATASET_DIR + '/' + actor_folder + '/' + wavfile
                    output_dir_path = OUTPUT_DIR + '/' + actor_folder
                    audio_utils.preprocess_audio(input_video_path, output_dir_path, self.SAMPLE_RATE, self.OFFSET, self.DURATION)
    
    def process_video(self):
        DATASET_DIR = self.DATASET_BASE_DIR + '/' + 'ravdess_speech_videos'
        OUTPUT_DIR = self.PREPROCESSED_VIDEO_SAVE_DIR
        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder.split('_')[1]
            print(act_no)
            for mp4file in os.listdir(DATASET_DIR + "/" + actor_folder ):

                if mp4file.endswith(".mp4") and mp4file.startswith('01'):
                    print(mp4file)

                    input_video_path = DATASET_DIR + '/' + actor_folder + '/' + mp4file
                    output_dir_path = OUTPUT_DIR + '/' + actor_folder
                    video_utils.preprocess_video(input_video_path, output_dir_path, 10)
    
    def load_audio_filenames(self, random_state, test_size):
        print("hello") # Written by Diksha
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_AUDIO_DIR
        print(base_path)
        for actor_folder in os.listdir(base_path):
            actor_path = base_path + '/' + actor_folder
            files_list = []
            for audio_file in os.listdir(actor_path):
                files_list.append(audio_file)
            files_list.sort()
            for audio_file in files_list:
                audio_path = actor_path + '/' + audio_file

                em_id = int(audio_file.split('-')[2]) - 1
                one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                # print(one_hot_em.shape)
                X[em_id].append(audio_path)
                Y[em_id].append(one_hot_em)

        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test
    
    def load_visual_filenames(self, random_state, test_size):
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_VIDEO_DIR
        print(base_path)
        for actor_folder in os.listdir(base_path):
            actor_path = base_path + '/' + actor_folder + '/' + 'subtracted_frames'
            files_list = []
            for image_file in os.listdir(actor_path):
                files_list.append(image_file)
            files_list.sort()
            for image_file in files_list:
                image_path = actor_path + '/' + image_file

                em_id = int(image_file.split('-')[2]) - 1
                one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                # print(one_hot_em)
                X[em_id].append(image_path)
                Y[em_id].append(one_hot_em)

        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test
    
    @property
    def emotion_classes(self):
        return self.EMOTION_CLASSES

    @property
    def audio_duration(self):
        return self.AUDIO_DURATION

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def offset(self):
        return self.OFFSET
        
class SAVEE(Preprocess):
    EMOTION_CLASSES = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
    EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    DURATION = 3
    OFFSET = 0
    SAMPLE_RATE = 441000

    def __init__(self):
        self.dataset_config = config.DatasetConfig("SAVEE")
        self.DATASET_BASE_DIR = self.dataset_config.DATASET_BASE_DIR
        self.PREPROCESSED_AUDIO_SAVE_DIR = self.dataset_config.PREPROCESSED_AUDIO_SAVE_DIR
        self.PREPROCESSED_AUDIO_DIR = self.dataset_config.PREPROCESSED_AUDIO_DIR
        self.PREPROCESSED_VIDEO_SAVE_DIR = self.dataset_config.PREPROCESSED_VIDEO_SAVE_DIR
        self.PREPROCESSED_VIDEO_DIR = self.dataset_config.PREPROCESSED_VIDEO_DIR
        self.MODEL_SAVE_DIR = self.dataset_config.MODEL_SAVE_DIR

    def extract_em_id(self, filename):
        filename = filename.split('.')[0]
        emotion_class = ""
        for ch in filename:
            if(ch.isdigit()):
                break
            emotion_class += ch
        return self.EMOTION_CLASSES.index(emotion_class)

    def process_audio(self):
        DATASET_DIR = self.DATASET_BASE_DIR + '/' + 'AudioData'
        OUTPUT_DIR = self.PREPROCESSED_AUDIO_SAVE_DIR
        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder
            print(act_no)
            for wavfile in os.listdir(DATASET_DIR + "/" + actor_folder):

                if wavfile.endswith(".wav"):
                    print(wavfile)

                    input_video_path = DATASET_DIR + '/' + actor_folder + '/' + wavfile
                    output_dir_path = OUTPUT_DIR + '/' + actor_folder
                    audio_utils.preprocess_audio(input_video_path, output_dir_path, self.SAMPLE_RATE, self.OFFSET, self.DURATION)
    
    def process_video(self):
        DATASET_DIR = self.DATASET_BASE_DIR + '/' + 'AudioVisualClip'
        OUTPUT_DIR = PREPROCESSED_VIDEO_SAVE_DIR
        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder
            print(act_no)
            for avifile in os.listdir(DATASET_DIR + "/" + actor_folder):

                if avifile.endswith(".avi"):
                    print(avifile)

                    input_video_path = DATASET_DIR + '/' + actor_folder + '/' + avifile
                    output_dir_path = OUTPUT_DIR + '/' + actor_folder
                    video_utils.preprocess_video(input_video_path, output_dir_path, 10)
    
    def load_audio_filenames(self, random_state, test_size):
        print("hello") # Written by Diksha
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_AUDIO_DIR
        print(base_path)
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

                em_id = self.extract_em_id(audio_file)
                one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                # print(one_hot_em)
                X[em_id].append(audio_path)
                Y[em_id].append(one_hot_em)
            
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test

    def load_visual_filenames(self, random_state, test_size):
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_VIDEO_DIR
        print(base_path)
        for actor_folder in os.listdir(base_path):
            actor_path = base_path + '/' + actor_folder + '/' + 'subtracted_frames'
            files_list = []
            for image_file in os.listdir(actor_path):
                files_list.append(image_file)
            files_list.sort()
            for image_file in files_list:
                image_path = actor_path + '/' + image_file

                em_id = self.extract_em_id(image_file)
                one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                # print(one_hot_em)
                X[em_id].append(image_path)
                Y[em_id].append(one_hot_em)
            
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test
    
    @property
    def emotion_classes(self):
        return self.EMOTION_CLASSES

    @property
    def audio_duration(self):
        return self.AUDIO_DURATION

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def offset(self):
        return self.OFFSET


class RML(Preprocess):
    EMOTION_CLASSES = ['an', 'di', 'fe', 'ha', 'sa', 'su']
    EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised']
    DURATION = 4
    OFFSET = 0
    SAMPLE_RATE = 22050

    def __init__(self):
        self.dataset_config = config.DatasetConfig("RML")
        self.DATASET_BASE_DIR = self.dataset_config.DATASET_BASE_DIR
        self.PREPROCESSED_AUDIO_SAVE_DIR = self.dataset_config.PREPROCESSED_AUDIO_SAVE_DIR
        self.PREPROCESSED_AUDIO_DIR = self.dataset_config.PREPROCESSED_AUDIO_DIR
        self.PREPROCESSED_VIDEO_SAVE_DIR = self.dataset_config.PREPROCESSED_VIDEO_SAVE_DIR
        self.PREPROCESSED_VIDEO_DIR = self.dataset_config.PREPROCESSED_VIDEO_DIR
        self.MODEL_SAVE_DIR = self.dataset_config.MODEL_SAVE_DIR

    def extract_em_id(self, filename):
        filename = filename.split('.')[0]
        emotion_class = ""
        for ch in filename:
            if(ch.isdigit()):
                break
            emotion_class += ch
        return self.EMOTION_CLASSES.index(emotion_class)

    def process_audio(self):
        DATASET_DIR = os.path.join(self.DATASET_BASE_DIR, 'AudioData')
        print("dataset",DATASET_DIR)
        OUTPUT_DIR = self.PREPROCESSED_AUDIO_SAVE_DIR
        print("output file for audio", self.PREPROCESSED_AUDIO_SAVE_DIR)

        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder
            print(act_no)
            for lang in os.listdir(os.path.join(DATASET_DIR, actor_folder)):
                for wavfile in os.listdir(os.path.join(DATASET_DIR, actor_folder, lang)):
                    if wavfile.endswith(".wav"):
                        print(wavfile)

                        input_video_path = os.path.join(DATASET_DIR, actor_folder, lang, wavfile)
                        output_dir_path = os.path.join(OUTPUT_DIR, actor_folder, lang)
                        audio_utils.preprocess_audio(input_video_path, output_dir_path, self.SAMPLE_RATE, self.OFFSET, self.DURATION)

    def process_video(self):
        print(self.DATASET_BASE_DIR)
        DATASET_DIR = os.path.join(self.DATASET_BASE_DIR, 'VideoData')
        OUTPUT_DIR = self.PREPROCESSED_VIDEO_SAVE_DIR
        for actor_folder in os.listdir(DATASET_DIR):
            print(actor_folder)
            act_no = actor_folder
            print(act_no)
            for lang in os.listdir(os.path.join(DATASET_DIR, actor_folder)):
                for avifile in os.listdir(os.path.join(DATASET_DIR, actor_folder, lang)):
                    if avifile.endswith(".avi"):
                        print(avifile)

                        input_video_path = os.path.join(DATASET_DIR, actor_folder, lang, avifile)
                        output_dir_path = os.path.join(OUTPUT_DIR, actor_folder, lang)
                        video_utils.preprocess_video(input_video_path, output_dir_path, 10)

    def load_audio_filenames(self, random_state, test_size):
        print("hello")
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_AUDIO_DIR
        print(base_path)
        for actor_folder in os.listdir(base_path):
            actor_path = os.path.join(base_path, actor_folder)
            for lang in os.listdir(os.path.join(actor_path)):
                actor_lang_path = os.path.join(actor_path, lang)
                files_list = []
                for audio_file in os.listdir(actor_lang_path):
                    files_list.append(audio_file)
                files_list.sort()
                for audio_file in files_list:
                    audio_path = os.path.join(actor_lang_path, audio_file)
                    # S_input = np.load(audio_path)
                    # # print(S_input.shape)
                    # X.append(S_input) # (216,1)

                    em_id = self.extract_em_id(audio_file)
                    one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                    # print(one_hot_em)
                    X[em_id].append(audio_path)
                    Y[em_id].append(one_hot_em)
            
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test

    def load_visual_filenames(self, random_state, test_size):
        X = []
        Y = []
        for i in range(len(self.EMOTION_CLASSES)):
            X.append([])
            Y.append([])
        base_path = self.PREPROCESSED_VIDEO_DIR
        print(base_path)
        for actor_folder in os.listdir(base_path):
            actor_path = os.path.join(base_path, actor_folder)
            for lang in os.listdir(actor_path):
                actor_lang_path = os.path.join(actor_path, lang, 'subtracted_frames')
                files_list = []
                for image_file in os.listdir(actor_lang_path):
                    files_list.append(image_file)
                files_list.sort()
                for image_file in files_list:
                    image_path = os.path.join(actor_lang_path, image_file)

                    em_id = self.extract_em_id(image_file)
                    one_hot_em = tf.one_hot(em_id, len(self.EMOTION_CLASSES))
                    # print(one_hot_em)
                    X[em_id].append(image_path)
                    Y[em_id].append(one_hot_em)
            
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for (file_list, label_list, em_id) in zip(X, Y, range(len(self.EMOTION_CLASSES))):
            train_test_distribution = train_test_split(file_list, label_list, test_size = test_size, 
                                        shuffle = True, random_state = random_state + em_id)
            X_train += train_test_distribution[0]
            X_test += train_test_distribution[1]
            Y_train += train_test_distribution[2]
            Y_test += train_test_distribution[3]
        
        X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train), random_state = random_state)
        X_test, Y_test = shuffle(np.array(X_test), np.array(Y_test), random_state = random_state)
        return X_train, X_test, Y_train, Y_test
    
    @property
    def emotion_classes(self):
        return self.EMOTION_CLASSES

    @property
    def audio_duration(self):
        return self.AUDIO_DURATION

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    @property
    def offset(self):
        return self.OFFSET

        