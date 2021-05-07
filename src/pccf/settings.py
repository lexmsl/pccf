import os


ROI_RADIUS_EARTH_QUAKES = 4
ROI_RADIUS_LAKE = 70
ROI_RADIUS_TWI = 7
ROI_RADIUS_ARTIFICIAL_SIGNAL = 25

CHANGES_TWI = [29, 50, 77, 133, 153, 175, 193, 223]
CHANGES_TWI = CHANGES_TWI[-3:]

PROJECT_DIR = "/home/ms314/1/phd"
TEX_FOLDER_IMG = os.path.join(PROJECT_DIR, "tex/img")  # "/home/ms314/1/phd/tex/img"
SRC_DIR = os.path.join(PROJECT_DIR, "src/pccf")
SRC_TEMP_DIR = os.path.join(SRC_DIR, 'temp')

SIG_HALF_LEN = 100

