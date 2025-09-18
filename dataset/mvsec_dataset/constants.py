MVSEC_HEIGHT = 260
MVSEC_WIDTH = 346

MVSEC_TRAIN = {
    "train/mvsec_dataset_day2/train":8524, "train/mvsec_dataset_day2/validation":1827, 
}

MVSEC_VALIDATION = {
    "train/mvsec_dataset_day2/test":1827, 
}

MVSEC_TEST = {
    "test/mvsec_outdoor_day1":5125, "test/mvsec_outdoor_night1":5111, "test/mvsec_outdoor_night2":5478, "test/mvsec_outdoor_night3":5411,
}

MVSEC_ALL_DATA_FOLDERS = list(MVSEC_TRAIN.keys()) + list(MVSEC_VALIDATION.keys()) + list(MVSEC_TEST.keys())