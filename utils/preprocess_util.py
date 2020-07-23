import os
import utils.local_config as local_config
import utils.audio_utils as audio_utils
import utils.video_utils as video_utils

DATA_DIR = local_config.DATA_DIR
PREPROCESSED_VIDEO_DIR = local_config.PREPROCESSED_VIDEO_DIR
PREPROCESSED_AUDIO_DIR = local_config.PREPROCESSED_AUDIO_DIR

def preprocess_ravdess_audio_data():
    DATASET_DIR = DATA_DIR + '/' + 'Audio_Speech_Actors_01-24'
    OUTPUT_DIR = PREPROCESSED_AUDIO_DIR
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

def preprocess_ravdess_facial_data():
    DATASET_DIR = DATA_DIR + '/' + 'ravdess_speech_videos'
    OUTPUT_DIR = PREPROCESSED_VIDEO_DIR
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

def preprocess_savee_audio_data():
    DATASET_DIR = DATA_DIR + '/' + 'AudioData'
    OUTPUT_DIR = PREPROCESSED_AUDIO_DIR
    for actor_folder in os.listdir(DATASET_DIR):
        print(actor_folder)
        act_no = actor_folder
        print(act_no)
        for wavfile in os.listdir(DATASET_DIR + "/" + actor_folder):

            if wavfile.endswith(".wav"):
                print(wavfile)

                input_video_path = DATASET_DIR + '/' + actor_folder + '/' + wavfile
                output_dir_path = OUTPUT_DIR + '/' + actor_folder
                audio_utils.preprocess_audio(input_video_path, output_dir_path)

def preprocess_savee_facial_data():
    DATASET_DIR = DATA_DIR + '/' + 'AudioVisualClip'
    OUTPUT_DIR = PREPROCESSED_VIDEO_DIR
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

