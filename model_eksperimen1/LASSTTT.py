# single_cell_multimodal_gpu.py
# Tempel dan jalankan di notebook lokal. Pastikan cell ini dieksekusi setelah install library.
import os
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import librosa
import pretty_midi
import pickle
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# Config & device
# -----------------------
os.makedirs('visualitation', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

def move_batch(batch, device):
    # batch is dict of tensors or tuple (we use dict in this notebook)
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def savefig(fname):
    path = os.path.join('visualitation', fname)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"{fname} disimpan di visualitation/")

# -----------------------
# Load dataset paths (keep your detection block)
# -----------------------
print("\n" + "="*40)
print("LOADING LOCAL DATASET")
print("="*40)

dataset_dir = None
candidate_paths = [
    os.path.abspath(os.path.join(os.getcwd(), '..', 'dataset')),
    os.path.abspath(os.path.join(os.getcwd(), 'dataset')),
    r'd:\Se\Multimodal\TUBES\dataset',
]
for p in candidate_paths:
    if os.path.exists(p):
        dataset_dir = p
        print("✓ Found dataset at:", dataset_dir)
        break
if dataset_dir is None:
    print("❌ Dataset not found. Using default path variable (edit if perlu).")
    dataset_dir = r'd:\Se\Multimodal\TUBES\dataset'

lyrics_dir = os.path.join(dataset_dir, 'Lyrics')
audio_dir = os.path.join(dataset_dir, 'Audio')
midi_dir = os.path.join(dataset_dir, 'MIDIs')
print("Validate: Lyrics", os.path.exists(lyrics_dir), "Audio", os.path.exists(audio_dir), "MIDI", os.path.exists(midi_dir))
if not (os.path.exists(lyrics_dir) and os.path.exists(audio_dir) and os.path.exists(midi_dir)):
    raise FileNotFoundError("Folder Lyrics/Audio/MIDIs tidak lengkap. Periksa dataset_dir.")

# -----------------------
# Load cluster labels (as in original)
# -----------------------
def load_cluster_labels(dataset_path):
    clusters_path = os.path.join(dataset_path, 'clusters.txt')
    if not os.path.exists(clusters_path):
        raise FileNotFoundError("clusters.txt tidak ditemukan di dataset.")
    with open(clusters_path, 'r', encoding='utf-8', errors='ignore') as f:
        cluster_labels = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(cluster_labels)} cluster labels")
    return cluster_labels

cluster_labels = load_cluster_labels(dataset_dir)
song_cluster_map = {}
for idx in range(len(cluster_labels)):
    for song_id in [str(idx).zfill(3), str(idx+1).zfill(3)]:
        song_cluster_map[song_id] = cluster_labels[idx]

# -----------------------
# Load BERT and PANNs (prefer local checkpoint)
# -----------------------
print("\nLoading BERT and PANNs (local preferred)...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()
print("✓ BERT ready")

# PANNs: try panns_inference AudioTagging, fallback to torch.hub skeleton with local .pth
panns_model = None
pann_ckpt_candidates = [
    r"D:\Se\Multimodal\TUBES\Cnn14_mAP%3D0.431.pth",
    r"D:\Se\Multimodal\TUBES\Cnn14_mAP=0.431.pth",
    os.path.join(dataset_dir, 'Cnn14_mAP%3D0.431.pth'),
    os.path.join(dataset_dir, 'Cnn14_mAP=0.431.pth'),
]
# try panns_inference
try:
    from panns_inference import AudioTagging
    # if user wants to use specific local file, pass it
    ck = next((c for c in pann_ckpt_candidates if os.path.exists(c)), None)
    if ck:
        panns_model = AudioTagging(checkpoint_path=ck, device=device)
        print("✓ panns_inference AudioTagging loaded from local checkpoint")
    else:
        try:
            panns_model = AudioTagging(device=device)  # try default pretrained
            print("✓ panns_inference AudioTagging loaded (pretrained)")
        except Exception:
            panns_model = None
except Exception as e:
    panns_model = None
    print("panns_inference not available or failed:", e)

# fallback: try torch.hub cnn14 skeleton + load local state_dict non-strict
if panns_model is None:
    ck = next((c for c in pann_ckpt_candidates if os.path.exists(c)), None)
    if ck:
        try:
            model_skel = torch.hub.load('qiuqiangkong/panns_audio', 'cnn14', pretrained=False)
            state = torch.load(ck, map_location='cpu')
            # get state dict
            if isinstance(state, dict):
                sd = state.get('model', state.get('state_dict', state))
            else:
                sd = state
            # normalize keys
            new_sd = {}
            if isinstance(sd, dict):
                for k, v in sd.items():
                    nk = k
                    if nk.startswith('module.'):
                        nk = nk[7:]
                    if nk.startswith('model.'):
                        nk = nk[len('model.'):]
                    new_sd[nk] = v
            else:
                new_sd = sd
            missing, unexpected = model_skel.load_state_dict(new_sd, strict=False)
            model_skel.eval()
            model_skel.to(device)
            panns_model = model_skel
            print("✓ cnn14 skeleton loaded and state_dict applied (non-strict).")
            print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
        except Exception as e:
            panns_model = None
            print("Gagal load cnn14 checkpoint:", e)
    else:
        # try torch.hub pretrained
        try:
            panns_model = torch.hub.load('qiuqiangkong/panns_audio', 'cnn14', pretrained=True)
            panns_model.eval()
            panns_model.to(device)
            print("✓ panns cnn14 loaded from torch.hub pretrained")
        except Exception as e:
            panns_model = None
            print("panns load gagal:", e)

if panns_model is None:
    print("PANNs unavailable. Audio will use MFCC fallback.")

# -----------------------
# Feature extraction helpers (lyrics, audio, midi)
# -----------------------
def clean_lyrics(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r'\[.*?\]|\(.*?\)|http\S+|www\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s.,!?\']', ' ', s)
    s = ' '.join(s.split())
    return s

def extract_lyrics_embedding(lyrics, tokenizer, model, max_length=256):
    try:
        s = clean_lyrics(lyrics)
        if not s or len(s) < 10:
            return None
        enc = tokenizer(s, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            last = out.last_hidden_state  # B L H
            attn_f = attn.float().unsqueeze(-1)
            emb = (last * attn_f).sum(dim=1).cpu().numpy()[0]
        return emb
    except Exception:
        return None

def augment_audio(audio, sr):
    try:
        if np.random.rand() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if np.random.rand() < 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        return audio
    except Exception:
        return audio

def extract_audio_embedding(audio_path, panns_model, sr=32000, duration=10, augment=False):
    try:
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if augment:
            audio = augment_audio(audio, sr)
        target_len = sr * duration
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        if panns_model is None:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            return np.mean(mfcc, axis=1)
        # panns_inference AudioTagging returns (clipwise_output, embedding)
        if hasattr(panns_model, 'inference'):
            _, emb = panns_model.inference(audio[None, :])
            return emb[0]
        else:
            # if torch model cnn14 expecting waveform tensor (B, samples)
            if isinstance(panns_model, torch.nn.Module):
                wav = torch.tensor(audio[None, :], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = panns_model(wav)
                if isinstance(out, (tuple, list)) and len(out) > 1:
                    emb = out[1]
                    return emb.detach().cpu().numpy().reshape(-1)
                else:
                    # try flatten
                    return np.array(out).reshape(-1)[:2048]
    except Exception:
        return None

def extract_midi_features(midi_path, augment=False, timeout=10):
    # simplified safe MIDI parse
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instr in midi.instruments:
            if not instr.is_drum:
                for n in instr.notes:
                    notes.append((n.pitch, n.velocity, n.end - n.start))
        if len(notes) == 0:
            return None
        pitches = np.array([n[0] for n in notes])
        velocities = np.array([n[1] for n in notes])
        durations = np.array([n[2] for n in notes])
        tempo_changes = midi.get_tempo_changes()
        avg_tempo = np.mean(tempo_changes[1]) if len(tempo_changes[1])>0 else 120.0
        features = np.array([
            np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches),
            np.percentile(pitches, 25), np.percentile(pitches, 75), np.ptp(pitches), len(set(pitches)),
            np.mean(velocities), np.std(velocities), np.min(velocities), np.max(velocities),
            np.percentile(velocities,25), np.percentile(velocities,75), np.ptp(velocities), len(notes),
            np.mean(durations), np.std(durations), np.min(durations), np.max(durations),
            np.percentile(durations,25), np.percentile(durations,75), np.ptp(durations), 1.0/(np.mean(durations)+1e-6),
            avg_tempo, avg_tempo/120.0, midi.get_end_time(), len(midi.instruments)
        ], dtype=np.float32)
        if augment:
            features = augment_midi_features(features)
        return features[:32] if features.shape[0] >= 32 else np.pad(features, (0, 32 - features.shape[0]))
    except Exception:
        return None

def augment_midi_features(features):
    f = features.copy()
    if np.random.rand() < 0.5:
        f += np.random.normal(0, 0.01, size=f.shape)
    return f

# -----------------------
# Build dataset list (same logic as you had)
# -----------------------
print("\nExtracting features from files (this may take time)...")
lyrics_files = {f.replace('.txt',''): f for f in os.listdir(lyrics_dir) if f.endswith('.txt')}
audio_files = {os.path.splitext(f)[0]: f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))}
midi_files = {os.path.splitext(f)[0]: f for f in os.listdir(midi_dir) if f.endswith(('.mid', '.midi'))}
print("Found:", len(lyrics_files), "lyrics,", len(audio_files), "audio,", len(midi_files), "midi")

all_song_ids = set()
for key in lyrics_files.keys():
    song_id = ''.join(filter(str.isdigit, key))
    if song_id:
        all_song_ids.add(song_id.zfill(3))
print("Processing", len(all_song_ids), "songs")

data_list = []
processed = 0
skipped = 0
for song_id in sorted(all_song_ids):
    if song_id not in song_cluster_map:
        skipped += 1
        continue
    lyrics_emb = None
    audio_emb = None
    midi_feat = None
    # lyrics
    for k, fn in lyrics_files.items():
        if song_id in k:
            try:
                with open(os.path.join(lyrics_dir, fn), 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                lyrics_emb = extract_lyrics_embedding(txt, bert_tokenizer, bert_model)
            except Exception:
                lyrics_emb = None
            break
    # audio
    for k, fn in audio_files.items():
        if song_id in k:
            try:
                audio_emb = extract_audio_embedding(os.path.join(audio_dir, fn), panns_model, augment=False)
            except Exception:
                audio_emb = None
            break
    # midi
    for k, fn in midi_files.items():
        if song_id in k:
            try:
                midi_feat = extract_midi_features(os.path.join(midi_dir, fn), augment=False)
            except Exception:
                midi_feat = None
            break
    available = sum([lyrics_emb is not None, audio_emb is not None, midi_feat is not None])
    if available >= 2:
        if audio_emb is not None:
            if audio_emb.shape[0] != 2048:
                if audio_emb.shape[0] < 2048:
                    audio_emb = np.pad(audio_emb, (0, 2048 - audio_emb.shape[0]))
                else:
                    audio_emb = audio_emb[:2048]
        data_list.append({
            'song_id': song_id,
            'lyrics_emb': lyrics_emb if lyrics_emb is not None else np.zeros(768, dtype=np.float32),
            'audio_emb': audio_emb if audio_emb is not None else np.zeros(2048, dtype=np.float32),
            'midi_feat': midi_feat if midi_feat is not None else np.zeros(32, dtype=np.float32),
            'has_lyrics': 1 if lyrics_emb is not None else 0,
            'has_audio': 1 if audio_emb is not None else 0,
            'has_midi': 1 if midi_feat is not None else 0,
            'cluster': song_cluster_map[song_id]
        })
        processed += 1

df = pd.DataFrame(data_list)
print("Feature extraction done. Samples:", len(df))

# -----------------------
# Labels and distribution
# -----------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['cluster'])
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_, "Num classes:", num_classes)

# -----------------------
# Dataset class and DataLoader
# -----------------------
class MultimodalDataset(Dataset):
    def __init__(self, df, augment=False, audio_files_map=None, midi_files_map=None):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.audio_files_map = audio_files_map
        self.midi_files_map = midi_files_map
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {
            'lyrics_emb': torch.FloatTensor(r['lyrics_emb']),
            'audio_emb': torch.FloatTensor(r['audio_emb']),
            'midi_feat': torch.FloatTensor(r['midi_feat']),
            'has_lyrics': torch.FloatTensor([r['has_lyrics']]),
            'has_audio': torch.FloatTensor([r['has_audio']]),
            'has_midi': torch.FloatTensor([r['has_midi']]),
            'label': torch.tensor(r['label'], dtype=torch.long)
        }

# -----------------------
# Model classes (keep as provided)
# -----------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*2, dim))
    def forward(self, query, key_value):
        attn_out, _ = self.multihead_attn(query, key_value, key_value)
        out = self.norm(query + attn_out)
        out = out + self.ffn(out)
        return out

class LightweightMIDIEncoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

class CrossAttentionMultimodalClassifier(nn.Module):
    def __init__(self, num_classes, lyrics_dim=768, audio_dim=2048, midi_dim=32, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.lyrics_proj = nn.Sequential(nn.Linear(lyrics_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout*0.5))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout*0.5))
        self.midi_encoder = LightweightMIDIEncoder(midi_dim, hidden_dim)
        self.cross_attn_la = CrossAttentionFusion(hidden_dim)
        self.cross_attn_lm = CrossAttentionFusion(hidden_dim)
        self.cross_attn_am = CrossAttentionFusion(hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*2), nn.LayerNorm(hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(dropout*0.5))
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, lyrics_emb, audio_emb, midi_feat, has_lyrics, has_audio, has_midi):
        lyrics_feat = self.lyrics_proj(lyrics_emb)
        audio_feat = self.audio_proj(audio_emb)
        midi_feat = self.midi_encoder(midi_feat)
        lyrics_feat = lyrics_feat * has_lyrics
        audio_feat = audio_feat * has_audio
        midi_feat = midi_feat * has_midi
        l = lyrics_feat.unsqueeze(1); a = audio_feat.unsqueeze(1); m = midi_feat.unsqueeze(1)
        la_fused = self.cross_attn_la(l, a).squeeze(1)
        lm_fused = self.cross_attn_lm(l, m).squeeze(1)
        am_fused = self.cross_attn_am(a, m).squeeze(1)
        fused = torch.cat([la_fused, lm_fused, am_fused], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# -----------------------
# Training helpers
# -----------------------
def train_epoch(model, bert_model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    if epoch >= 3:
        bert_model.train()
        for p in bert_model.parameters():
            p.requires_grad = True
    else:
        bert_model.eval()
    total_loss = 0
    preds_all = []; labels_all = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        logits = model(batch['lyrics_emb'], batch['audio_emb'], batch['midi_feat'],
                       batch['has_lyrics'], batch['has_audio'], batch['has_midi'])
        loss = criterion(logits, batch['label'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if epoch >= 3:
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.cpu().numpy()); labels_all.extend(batch['label'].cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(labels_all, preds_all)

def evaluate(model, bert_model, dataloader, criterion, device):
    model.eval(); bert_model.eval()
    total_loss = 0; preds_all = []; labels_all = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            logits = model(batch['lyrics_emb'], batch['audio_emb'], batch['midi_feat'],
                           batch['has_lyrics'], batch['has_audio'], batch['has_midi'])
            loss = criterion(logits, batch['label'])
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy()); labels_all.extend(batch['label'].cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='weighted', zero_division=0)
    return total_loss/len(dataloader), accuracy_score(labels_all, preds_all), p, r, f1, preds_all, labels_all

# -----------------------
# 5-Fold CV training (cleaned)
# -----------------------
BATCH_SIZE = 16 if device.type == 'cuda' else 8
NUM_WORKERS = 0
EPOCHS = 30
PATIENCE = 8
WARMUP_RATIO = 0.1

X = df.index.values; y = df['label'].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []; all_histories = {}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== FOLD {fold+1}/5 ===")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_labels = train_df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_ds = MultimodalDataset(train_df, augment=True, audio_files_map=audio_files, midi_files_map=midi_files)
    val_ds = MultimodalDataset(val_df, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model = CrossAttentionMultimodalClassifier(num_classes=num_classes).to(device)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = AdamW([{'params': model.classifier.parameters(), 'lr':5e-4},
                       {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr':1e-5}], weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_precision':[], 'val_recall':[], 'val_f1':[], 'epochs':[]}
    best_f1 = 0; patience_cnt = 0
    for epoch in range(EPOCHS):
        try:
            train_loss, train_acc = train_epoch(model, bert_model, train_loader, criterion, optimizer, scheduler, device, epoch)
            val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(model, bert_model, val_loader, criterion, device)
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['val_precision'].append(val_p); history['val_recall'].append(val_r); history['val_f1'].append(val_f1)
            history['epochs'].append(epoch+1)
            print(f"Epoch {epoch+1}/{EPOCHS} TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} ValF1 {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({'model':model.state_dict(), 'bert':bert_model.state_dict(), 'history':history}, f'best_fold{fold+1}.pt')
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print("Early stopping")
                    break
        except Exception as e:
            print("Error epoch:", e)
            break
    with open(f'history_fold{fold+1}.pkl','wb') as f:
        pickle.dump(history, f)
    all_histories[f'fold_{fold+1}'] = history
    try:
        ck = torch.load(f'best_fold{fold+1}.pt', map_location=device)
        model.load_state_dict(ck['model'])
        bert_model.load_state_dict(ck['bert'])
        val_loss, val_acc, val_p, val_r, val_f1, preds, labels = evaluate(model, bert_model, val_loader, criterion, device)
        print(f"Fold {fold+1} result Acc {val_acc:.4f} F1 {val_f1:.4f}")
        print(classification_report(labels, preds, target_names=label_encoder.classes_, digits=4, zero_division=0))
        fold_results.append({'fold':fold+1,'accuracy':val_acc,'precision':val_p,'recall':val_r,'f1':val_f1})
    except Exception as e:
        print("Could not load best checkpoint:", e)

# save combined histories
with open('all_training_histories.pkl','wb') as f:
    pickle.dump(all_histories, f)
print("Training finished. Histories saved.")

# -----------------------
# Visualization & Metrics
# -----------------------
print("\nGenerating visualizations...")

# Confusion matrices averaged across folds
all_true = []; all_pred = []; fold_cms = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X,y)):
    ck_path = f'best_fold{fold_idx+1}.pt'
    if not os.path.exists(ck_path):
        print("Checkpoint missing for fold", fold_idx+1); continue
    ck = torch.load(ck_path, map_location=device)
    model = CrossAttentionMultimodalClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_ds = MultimodalDataset(val_df, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    preds_fold = []; labels_fold = []
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            logits = model(batch['lyrics_emb'], batch['audio_emb'], batch['midi_feat'],
                           batch['has_lyrics'], batch['has_audio'], batch['has_midi'])
            preds = torch.argmax(logits, dim=1)
            preds_fold.extend(preds.cpu().numpy()); labels_fold.extend(batch['label'].cpu().numpy())
    all_true.extend(labels_fold); all_pred.extend(preds_fold)
    cm = confusion_matrix(labels_fold, preds_fold, labels=list(range(num_classes)))
    fold_cms.append(cm)

if len(fold_cms) > 0:
    avg_cm = np.mean(fold_cms, axis=0)
    plt.figure(figsize=(10,8))
    plt.imshow(avg_cm, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title('Average Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(np.arange(num_classes), label_encoder.classes_, rotation=90)
    plt.yticks(np.arange(num_classes), label_encoder.classes_)
    for i in range(avg_cm.shape[0]):
        for j in range(avg_cm.shape[1]):
            plt.text(j, i, int(avg_cm[i,j]), ha='center', va='center', color='white' if avg_cm[i,j]>avg_cm.max()/2 else 'black', fontsize=8)
    savefig('confusion_matrix_avg.png')
else:
    print("No confusion matrices to plot")

# Classification report overall
if len(all_true) > 0:
    print("\nClassification Report (overall):")
    print(classification_report(all_true, all_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))
    # save as text
    with open('visualitation/classification_report.txt','w') as f:
        f.write(classification_report(all_true, all_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))
    print("Classification report saved: visualitation/classification_report.txt")

# MAE
if len(all_true) > 0:
    mae = mean_absolute_error(all_true, all_pred)
    print("MAE:", mae)
    # histogram error distribution
    err = np.abs(np.array(all_true) - np.array(all_pred))
    plt.figure(figsize=(8,5))
    plt.hist(err, bins=np.arange(num_classes+1)-0.5, edgecolor='black')
    plt.xlabel('Absolute error (classes)'); plt.ylabel('Frequency')
    plt.title(f'MAE: {mae:.4f}')
    savefig('mae_distribution.png')

# Learning curves from histories
try:
    with open('all_training_histories.pkl','rb') as f:
        all_hist = pickle.load(f)
    # average curves
    max_epochs = max(len(h['epochs']) for h in all_hist.values())
    avg_train_loss = np.zeros(max_epochs); avg_val_loss = np.zeros(max_epochs)
    avg_train_acc = np.zeros(max_epochs); avg_val_acc = np.zeros(max_epochs); counts = np.zeros(max_epochs)
    for h in all_hist.values():
        for i, e in enumerate(h['epochs']):
            avg_train_loss[i] += h['train_loss'][i]; avg_val_loss[i] += h['val_loss'][i]
            avg_train_acc[i] += h['train_acc'][i]; avg_val_acc[i] += h['val_acc'][i]
            counts[i] += 1
    idx = counts>0
    epochs = np.arange(1, int(idx.sum())+1)
    avg_train_loss = (avg_train_loss[idx] / counts[idx])
    avg_val_loss = (avg_val_loss[idx] / counts[idx])
    avg_train_acc = (avg_train_acc[idx] / counts[idx])
    avg_val_acc = (avg_val_acc[idx] / counts[idx])
    plt.figure(figsize=(8,5))
    plt.plot(epochs, avg_train_loss, label='Train Loss'); plt.plot(epochs, avg_val_loss, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Average Loss Curves')
    savefig('learning_curve_loss_avg.png')
    plt.figure(figsize=(8,5))
    plt.plot(epochs, avg_train_acc, label='Train Acc'); plt.plot(epochs, avg_val_acc, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Average Accuracy Curves')
    savefig('learning_curve_acc_avg.png')
    print("Learning curves saved.")
except Exception as e:
    print("Could not plot learning curves:", e)

# Save summary CSV
if len(fold_results) > 0:
    res_df = pd.DataFrame(fold_results)
    res_df.to_csv('visualitation/enhanced_multimodal_results.csv', index=False)
    print("Fold results saved: visualitation/enhanced_multimodal_results.csv")

print("\nSelesai. Semua file visual disimpan di folder visualitation.")
