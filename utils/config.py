from configparser import ConfigParser, ExtendedInterpolation


config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('./utils/config.ini')

global_config = config['GLOBAL']

# global config
BASE_DIR = global_config['base_dir']
HYPARAM_CONFIG_FILE_PATH = global_config['hyparam_config_file_path']


class DatasetConfig:

    def __init__(self, dataset_name):

        # dataset config
        self.dataset_config = config[dataset_name]
        if(self.dataset_config is None):
            raise AttributeError("invalid dataset_name")

        self.DATASET_BASE_DIR = self.dataset_config['dataset_base_dir']
        self.MODEL_SAVE_DIR = self.dataset_config['model_save_dir']

        self.DATA_DIR = self.dataset_config['data_dir']
        self.PREPROCESSED_VIDEO_DIR = self.dataset_config['preprocessed_video_dir']
        self.PREPROCESSED_AUDIO_DIR = self.dataset_config['preprocessed_audio_dir']

        self.DATA_SAVE_DIR = self.dataset_config['data_save_dir']
        self.PREPROCESSED_VIDEO_SAVE_DIR = self.dataset_config['preprocessed_video_save_dir']
        self.PREPROCESSED_AUDIO_SAVE_DIR = self.dataset_config['preprocessed_audio_save_dir']

