# config2.py
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

# Dataset paths
DATASET_ROOT = r"D:\Se\Multimodal\TUBES\dataset"
AUDIO_DIR = os.path.join(DATASET_ROOT, 'Audio')
LYRICS_DIR = os.path.join(DATASET_ROOT, 'Lyrics')
MIDI_DIR = os.path.join(DATASET_ROOT, 'MIDIs')
CLUSTERS_PATH = os.path.join(DATASET_ROOT, 'clusters.txt')

# Cache directory
CACHE_DIR = os.path.join(ROOT, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, 'dataset_embeddings.pkl')

# Model / training params
BATCH_SIZE = 16
LR_HEAD = 5e-4              # Learning rate for fusion model
LR_BACKBONE = 1e-5          # Learning rate for BERT (when fine-tuning)
EPOCHS = 30
PATIENCE = 8                # Early stopping patience
WARMUP_RATIO = 0.1          # Warmup ratio for scheduler
SEED = 42

# Feature dimensions
LYRICS_DIM = 768            # BERT output dimension
AUDIO_DIM = 2048            # PANN output (or MFCC fallback)
MIDI_DIM = 32               # MIDI feature vector dimension

# Additional MIDI params (for sequence-based processing)
MIDI_FEATURES = 4           # pitch, duration, velocity, time_delta
MIDI_SEQ_LEN = 512          # sequence length
MIDI_OUT_DIM = 128          # encoder output dimension

# Fusion model architecture
FUSION_HIDDEN = 512         # Hidden dimension for fusion layers
NUM_HEADS = 8               # Number of attention heads
DROPOUT = 0.2               # Dropout rate
NUM_TRANSFORMER_LAYERS = 2  # Number of transformer blocks after fusion

# Hierarchical classifier dimensions
CLASSIFIER_HIDDEN_DIMS = [1024, 512, 256]  # Progressive dimension reduction

# Audio processing params
SR = 32000                  # Sample rate
AUDIO_DURATION = 10         # Audio duration in seconds

# DataLoader params
DEFAULT_NUM_WORKERS = 2

# Model checkpoints
PANN_CHECKPOINT = r"D:\Se\Multimodal\TUBES\Cnn14_mAP%3D0.431.pth"

# Data augmentation
AUGMENT_AUDIO_PER_SAMPLE = 0   # Number of audio augmentations per sample
AUGMENT_MIDI_PER_SAMPLE = 0    # Number of MIDI augmentations per sample

# Tokenization
MAX_LYRIC_TOKENS = 256      # Maximum tokens for lyrics

# Loss function params
FOCAL_GAMMA = 2.0           # Focal loss gamma parameter
LABEL_SMOOTHING = 0.1       # Label smoothing factor
AUX_LOSS_WEIGHT = 0.3       # Weight for auxiliary loss

# Training strategies
FINE_TUNE_BERT_AFTER = 3    # Start fine-tuning BERT after N epochs
GRADIENT_CLIP_NORM = 1.0    # Gradient clipping norm
USE_MIXED_PRECISION = True  # Use automatic mixed precision (AMP)

# Model selection
# Options: 'advanced' (AdvancedMultimodalFusion) or 'improved' (ImprovedCrossAttentionFusion)
DEFAULT_MODEL_TYPE = 'advanced'