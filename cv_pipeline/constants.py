CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "Qu", "Er", "Th", "In", "An", "He", "BLOCK",
]

BOARD_CLASSES = {65, 66, 67}  # COCO class IDs: remote, keyboard, cell phone
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
TARGET_TILE_SIZE = 100
GRID_INSET_RATIO = 0.10
WARP_PAD_PCT = 0.07  # how far outside the quad corners to include in the warp
WARPED_IMG_MAX_SIZE = 480  # max side-length (px) for the warped board JPEG returned to frontend
WARPED_IMG_JPEG_QUALITY = 80
