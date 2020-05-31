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
