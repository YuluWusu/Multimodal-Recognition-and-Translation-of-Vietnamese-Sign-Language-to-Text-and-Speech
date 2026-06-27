import os

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

# ==============================================================================
# CẤU HÌNH MEDIAPIPE & OPENCV
# ==============================================================================
# Kích thước khung hình từ camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Cấu hình phát hiện và theo dõi MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ==============================================================================
# CẤU HÌNH HUẤN LUYỆN (TRAINING HYPERPARAMETERS)
# ==============================================================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


SEQUENCE_LENGTH = 60 

# ==============================================================================
# CẤU HÌNH TỪ VỰNG (VOCABULARY)
# ==============================================================================
# load từ file text, tạm thời để mảng
ACTIONS = [
    "địa chỉ (bắc)",
    "tỉnh",
    "bạn",
    
]
NUM_CLASSES = len(ACTIONS)
