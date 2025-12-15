# prepare_dataset.py
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
import librosa
import pretty_midi

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from config import (
    AUDIO_DIR, LYRICS_DIR, MIDI_DIR, CLUSTERS_PATH, CACHE_PATH,
    SR, AUDIO_DURATION, PANN_CHECKPOINT, MAX_LYRIC_TOKENS,
    AUGMENT_AUDIO_PER_SAMPLE, AUGMENT_MIDI_PER_SAMPLE,
    AUDIO_DIM, LYRICS_DIM, MIDI_DIM  # MIDI_DIM sekarang = 128
)

# ---------------------------
# Try load PANNs (optional)
# ---------------------------
USE_PANN = False
panns_model = None
try:
    from panns_inference import AudioTagging
    if os.path.exists(PANN_CHECKPOINT):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        panns_model = AudioTagging(checkpoint_path=PANN_CHECKPOINT, device=device_str)
        USE_PANN = True
        print("✓ PANNs loaded:", PANN_CHECKPOINT)
    else:
        print("⚠️ PANN checkpoint not found, fallback to MFCC")
except Exception as e:
    print("⚠️ panns_inference not available, fallback to MFCC:", e)

# ---------------------------
# Load BERT for lyrics
# ---------------------------
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(bert_device)
bert_model.eval()

# ---------------------------
# NEW: MIDI CNN Encoder (for emotion-aware piano roll embedding)
# ---------------------------
import torch.nn as nn

class MIDICNN128(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # (128, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.eval()

    def forward(self, x):
        return self.fc(self.conv(x))

# Initialize MIDI CNN once
_midi_cnn_model = None
_midi_cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _get_midi_cnn():
    global _midi_cnn_model
    if _midi_cnn_model is None:
        _midi_cnn_model = MIDICNN128(output_dim=MIDI_DIM).to(_midi_cnn_device)
    return _midi_cnn_model

def midi_to_pianoroll(midi_path, fs=50, max_time=30.0):
    """Convert MIDI to piano roll of shape (T, 128)"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        merged = pretty_midi.Instrument(program=0)
        for inst in pm.instruments:
            if not inst.is_drum:
                merged.notes.extend(inst.notes)
        if not merged.notes:
            return None
        pr = merged.get_piano_roll(fs=fs)  # (128, T)
        target_steps = int(fs * max_time)
        if pr.shape[1] < target_steps:
            pr = np.pad(pr, ((0, 0), (0, target_steps - pr.shape[1])), constant_values=0)
        else:
            pr = pr[:, :target_steps]
        return pr.T.astype(np.float32)  # (T, 128)
    except Exception:
        return None

def extract_midi_features(midi_path, timeout=10):
    """Extract 128-dim MIDI embedding using CNN on piano roll."""
    def _process(path):
        pr = midi_to_pianoroll(path, fs=50, max_time=30.0)
        if pr is None:
            return None
        x = torch.from_numpy(pr[None, None, :, :]).to(_midi_cnn_device)
        model = _get_midi_cnn()
        with torch.no_grad():
            emb = model(x)  # (1, MIDI_DIM)
        emb = emb[0].cpu().numpy().astype(np.float32)
        # Ensure correct dimension (should be MIDI_DIM=128)
        if emb.shape[0] != MIDI_DIM:
            if emb.shape[0] < MIDI_DIM:
                emb = np.pad(emb, (0, MIDI_DIM - emb.shape[0]))
            else:
                emb = emb[:MIDI_DIM]
        return emb

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_process, midi_path)
            return fut.result(timeout=timeout)
    except Exception:
        return None

# ---------------------------
# Helpers for lyrics and audio (unchanged)
# ---------------------------
def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r'\[.*?\]|\(.*?\)', ' ', s)
    s = re.sub(r'http\S+|www\S+', ' ', s)
    s = re.sub(r'[^\x00-\x7f]', ' ', s)
    s = re.sub(r'[^a-zA-Z0-9\s.,!?\']', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_lyrics(text, max_length=MAX_LYRIC_TOKENS):
    text = clean_text(text)
    if len(text) < 3:
        return None, None
    enc = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return enc['input_ids'][0].numpy(), enc['attention_mask'][0].numpy()

def extract_lyrics_pooled(text):
    text = clean_text(text)
    if len(text) < 3:
        return None
    try:
        enc = tokenizer(text, max_length=MAX_LYRIC_TOKENS, padding='max_length', truncation=True, return_tensors='pt')
        enc = {k: v.to(bert_device) for k, v in enc.items()}
        with torch.no_grad():
            out = bert_model(**enc)
            last = out.last_hidden_state
            att = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (last * att).sum(dim=1) / att.sum(dim=1).clamp(min=1.0)
            vec = pooled[0].cpu().numpy()
            if vec.shape[0] != LYRICS_DIM:
                if vec.shape[0] < LYRICS_DIM:
                    vec = np.pad(vec, (0, LYRICS_DIM - vec.shape[0]))
                else:
                    vec = vec[:LYRICS_DIM]
            return vec.astype(np.float32)
    except Exception:
        return None

def augment_audio_signal(audio, sr):
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        try:
            audio = librosa.effects.time_stretch(audio, rate)
        except Exception:
            pass
    if np.random.rand() < 0.5:
        n_steps = np.random.randint(-2, 3)
        try:
            audio = librosa.effects.pitch_shift(audio, sr, n_steps)
        except Exception:
            pass
    if np.random.rand() < 0.3:
        noise = np.random.randn(len(audio)) * 0.003
        audio = audio + noise
    return audio

def extract_audio_embedding(audio_path, sr=SR, duration=AUDIO_DURATION, augment=False):
    try:
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        target = sr * duration
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        if augment:
            audio = augment_audio_signal(audio, sr)
        if USE_PANN and panns_model is not None:
            try:
                with torch.no_grad():
                    _, emb = panns_model.inference(audio[None, :])
                emb = emb[0]
                emb = np.asarray(emb, dtype=np.float32)
            except Exception:
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                emb = np.mean(mfcc, axis=1).astype(np.float32)
        else:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            emb = np.mean(mfcc, axis=1).astype(np.float32)
        if emb is None:
            return None
        if emb.shape[0] < AUDIO_DIM:
            emb = np.pad(emb, (0, AUDIO_DIM - emb.shape[0]))
        else:
            emb = emb[:AUDIO_DIM]
        return emb.astype(np.float32)
    except Exception:
        return None

# ---------------------------
# Load cluster labels
# ---------------------------
if not os.path.exists(CLUSTERS_PATH):
    raise FileNotFoundError("clusters.txt not found at " + CLUSTERS_PATH)

cluster_labels = [line.strip() for line in open(CLUSTERS_PATH, 'r', encoding='utf-8', errors='ignore') if line.strip()]
song_cluster_map = {}
for idx in range(len(cluster_labels)):
    for song_id in [str(idx).zfill(3), str(idx+1).zfill(3)]:
        song_cluster_map[song_id] = cluster_labels[idx]

# ---------------------------
# Scan dataset files
# ---------------------------
lyrics_files = {os.path.splitext(f)[0]: f for f in os.listdir(LYRICS_DIR) if f.endswith('.txt')}
audio_files = {os.path.splitext(f)[0]: f for f in os.listdir(AUDIO_DIR) if f.endswith(('.mp3', '.wav'))}
midi_files = {os.path.splitext(f)[0]: f for f in os.listdir(MIDI_DIR) if f.endswith(('.mid', '.midi'))}

song_ids = sorted(set(list(lyrics_files.keys()) + list(audio_files.keys()) + list(midi_files.keys())))
print(f"Found {len(song_ids)} song ids")

rows = []
skipped = 0
midi_timeouts = []

for sid in tqdm(song_ids):
    if sid not in song_cluster_map:
        skipped += 1
        continue

    lyric_emb = None
    lyric_input_ids = None
    lyric_attn = None
    audio_emb = None
    midi_feat = None

    # lyrics
    if sid in lyrics_files:
        p = os.path.join(LYRICS_DIR, lyrics_files[sid])
        try:
            txt = open(p, 'r', encoding='utf-8', errors='ignore').read()
            lyric_input_ids, lyric_attn = tokenize_lyrics(txt)
            lyric_emb = extract_lyrics_pooled(txt)
        except Exception:
            pass

    # audio
    if sid in audio_files:
        p = os.path.join(AUDIO_DIR, audio_files[sid])
        try:
            audio_emb = extract_audio_embedding(p, augment=False)
        except Exception:
            pass

    # midi
    if sid in midi_files:
        p = os.path.join(MIDI_DIR, midi_files[sid])
        try:
            midi_feat = extract_midi_features(p, timeout=8)
            if midi_feat is None:
                midi_timeouts.append(sid)
        except Exception:
            pass

    available = sum([int(lyric_emb is not None), int(audio_emb is not None), int(midi_feat is not None)])
    if available < 2:
        continue

    # pad defaults
    if audio_emb is None:
        audio_emb = np.zeros(AUDIO_DIM, dtype=np.float32)
    if lyric_emb is None:
        lyric_emb = np.zeros(LYRICS_DIM, dtype=np.float32)
    if midi_feat is None:
        midi_feat = np.zeros(MIDI_DIM, dtype=np.float32)

    label = song_cluster_map.get(sid, None)
    if label is None:
        continue

    rows.append({
        'song_id': sid,
        'lyrics_emb': lyric_emb.astype(np.float32),
        'lyrics_input_ids': lyric_input_ids.astype(np.int64) if lyric_input_ids is not None else np.zeros(MAX_LYRIC_TOKENS, dtype=np.int64),
        'lyrics_attention_mask': lyric_attn.astype(np.int64) if lyric_attn is not None else np.zeros(MAX_LYRIC_TOKENS, dtype=np.int64),
        'audio_emb': audio_emb.astype(np.float32),
        'midi_feat': midi_feat.astype(np.float32),
        'has_lyrics': 1 if lyric_emb is not None else 0,
        'has_audio': 1 if audio_emb is not None else 0,
        'has_midi': 1 if midi_feat is not None else 0,
        'cluster': label
    })

print(f"Processed rows: {len(rows)}, skipped: {skipped}, midi_timeouts: {len(midi_timeouts)}")

if len(rows) == 0:
    raise SystemExit("No multimodal samples found (need >=2 modalities per sample)")

df = pd.DataFrame(rows)

# label encode
le = LabelEncoder()
df['label'] = le.fit_transform(df['cluster'])

# ---------------------------
# SMOTE for MIDI (only rows with midi)
# ---------------------------
midi_rows = df.loc[df['has_midi'] == 1].copy()
if len(midi_rows) > 10:
    print("Applying SMOTE on MIDI features...")
    X_midi = np.stack(midi_rows['midi_feat'].values)
    y_midi = midi_rows['label'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_midi)
    try:
        min_class_count = np.bincount(y_midi).min()
        k_neighbors = max(1, min(3, min_class_count - 1))
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        Xr, yr = sm.fit_resample(Xs, y_midi)
        Xr = scaler.inverse_transform(Xr)
        synthetic = []
        for xi, yi in zip(Xr, yr):
            synthetic.append({
                'song_id': f"smote_{len(df) + len(synthetic)}",
                'lyrics_emb': np.zeros(LYRICS_DIM, dtype=np.float32),
                'lyrics_input_ids': np.zeros(MAX_LYRIC_TOKENS, dtype=np.int64),
                'lyrics_attention_mask': np.zeros(MAX_LYRIC_TOKENS, dtype=np.int64),
                'audio_emb': np.zeros(AUDIO_DIM, dtype=np.float32),
                'midi_feat': xi.astype(np.float32),
                'has_lyrics': 0,
                'has_audio': 0,
                'has_midi': 1,
                'cluster': le.inverse_transform([int(yi)])[0],
                'label': int(yi)
            })
        if len(synthetic) > 0:
            df = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
            print(f"SMOTE added {len(synthetic)} synthetic MIDI samples")
    except Exception as e:
        print("SMOTE failed:", e)

# ---------------------------
# Optional augmentations
# ---------------------------
augmented = []
if AUGMENT_AUDIO_PER_SAMPLE > 0 or AUGMENT_MIDI_PER_SAMPLE > 0:
    print("Generating augmentations...")
    for idx, row in df.iterrows():
        sid = row['song_id']
        # audio augment
        if row['has_audio'] and AUGMENT_AUDIO_PER_SAMPLE > 0 and sid in audio_files:
            p = os.path.join(AUDIO_DIR, audio_files.get(sid, ''))
            try:
                audio, _ = librosa.load(p, sr=SR, duration=AUDIO_DURATION)
                for k in range(AUGMENT_AUDIO_PER_SAMPLE):
                    a_aug = augment_audio_signal(audio, SR)
                    if USE_PANN and panns_model is not None:
                        try:
                            _, emb = panns_model.inference(a_aug[None, :])
                            emb = emb[0]
                        except Exception:
                            mfcc = librosa.feature.mfcc(y=a_aug, sr=SR, n_mfcc=40)
                            emb = np.mean(mfcc, axis=1)
                    else:
                        mfcc = librosa.feature.mfcc(y=a_aug, sr=SR, n_mfcc=40)
                        emb = np.mean(mfcc, axis=1)
                    emb = np.asarray(emb, dtype=np.float32)
                    if emb.shape[0] < AUDIO_DIM:
                        emb = np.pad(emb, (0, AUDIO_DIM - emb.shape[0]))
                    else:
                        emb = emb[:AUDIO_DIM]
                    new = row.copy()
                    new['song_id'] = f"{sid}_augA{k}"
                    new['audio_emb'] = emb.astype(np.float32)
                    augmented.append(new)
            except Exception:
                pass
        # midi augment: add small noise to embedding
        if row['has_midi'] and AUGMENT_MIDI_PER_SAMPLE > 0:
            for k in range(AUGMENT_MIDI_PER_SAMPLE):
                aug = row['midi_feat'].copy()
                noise = np.random.normal(0, 0.02, size=aug.shape)
                aug = aug + noise
                new = row.copy()
                new['song_id'] = f"{sid}_augM{k}"
                new['midi_feat'] = aug.astype(np.float32)
                augmented.append(new)

if len(augmented) > 0:
    df_aug = pd.DataFrame(augmented)
    df = pd.concat([df, df_aug], ignore_index=True)
    print(f"Added {len(df_aug)} augmented samples")

# ---------------------------
# Save cache
# ---------------------------
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(CACHE_PATH, 'wb') as fh:
    pickle.dump({'df': df, 'label_encoder': le}, fh)

print("Saved cache:", CACHE_PATH)
print("Final dataset size:", len(df))