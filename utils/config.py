from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('./utils/config.ini')

global_config = config['GLOBAL']

# global config
BASE_DIR = global_config['base_dir']
HYPARAM_CONFIG_FILE_PATH = global_config['hyparam_config_file_path']
CURRENT_DATASET = global_config['current_dataset']


# dataset config
dataset_config = config[CURRENT_DATASET]
DATASET_BASE_DIR = dataset_config['dataset_base_dir']
MODEL_SAVE_DIR = dataset_config['model_save_dir']

DATA_DIR = dataset_config['data_dir']
PREPROCESSED_VIDEO_DIR = dataset_config['preprocessed_video_dir']
PREPROCESSED_AUDIO_DIR = dataset_config['preprocessed_audio_dir']

DATA_SAVE_DIR = dataset_config['data_save_dir']
PREPROCESSED_VIDEO_SAVE_DIR = dataset_config['preprocessed_video_save_dir']
PREPROCESSED_AUDIO_SAVE_DIR = dataset_config['preprocessed_audio_save_dir']
