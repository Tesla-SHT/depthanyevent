DSEC_HEIGHT = 480
DSEC_WIDTH = 640

DSEC_TRAIN = {
    "interlaken_00_c":269,  "interlaken_00_d":996,   "interlaken_00_e":996,  "zurich_city_00_a":470,
    "zurich_city_00_b":732, "zurich_city_01_a":341,  "zurich_city_01_b":663, "zurich_city_01_c":489,
    "zurich_city_01_d":398, "zurich_city_01_e":996,  "zurich_city_01_f":787, "zurich_city_02_a":118,
    "zurich_city_02_b":613, "zurich_city_02_c":1442, "zurich_city_02_d":922, "zurich_city_02_e":923,
    "zurich_city_03_a":442, "zurich_city_04_a":351,  "zurich_city_04_b":135, "zurich_city_04_c":591,
    "zurich_city_04_d":479, "zurich_city_04_e":135,  "zurich_city_04_f":430, "zurich_city_09_a":907,
    "zurich_city_09_b":184, "zurich_city_09_c":662,  "zurich_city_09_e":409, "zurich_city_10_a":1158,
    "zurich_city_11_a":233, "zurich_city_11_b":967,  "zurich_city_11_c":979,
}

DSEC_VALIDATION = {
    "interlaken_00_f":746,  "interlaken_00_g":668,   "thun_00_a":120,        "zurich_city_05_a":877,
    "zurich_city_05_b":815, "zurich_city_06_a":762,  "zurich_city_07_a":732, "zurich_city_08_a":394,
    "zurich_city_09_d":850, "zurich_city_10_b":1203,
}

DSEC_ALL_DATA_FOLDERS = list(DSEC_TRAIN.keys()) + list(DSEC_VALIDATION.keys())