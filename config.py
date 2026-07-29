import os
import json

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Đường dẫn file dữ liệu
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'f1_train.npz')
VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'f1_val.npz')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'f1_test.npz')
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, 'label_map.json')

# ==============================================================================
# CẤU HÌNH MEDIAPIPE & OPENCV
# ==============================================================================
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ==============================================================================
# CẤU HÌNH HUẤN LUYỆN (TRAINING HYPERPARAMETERS)
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 60

# ==============================================================================
# CẤU HÌNH TỪ VỰNG (VOCABULARY)
# ==============================================================================
NUM_CLASSES = 400

# Tự động load số lượng class từ label_map nếu file tồn tại
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        NUM_CLASSES = len(label_map)
        ACTIONS = list(label_map.keys()) if isinstance(label_map, dict) else label_map
else:
    ACTIONS = [] # Dự phòng
