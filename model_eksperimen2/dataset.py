# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal music classification
    """
    def __init__(self, dataframe):
        """
        Args:
            dataframe: pandas DataFrame with columns:
                - lyrics_emb: pooled lyrics embeddings (LYRICS_DIM)
                - lyrics_input_ids: tokenized lyrics (MAX_LYRIC_TOKENS)
                - lyrics_attention_mask: attention mask (MAX_LYRIC_TOKENS)
                - audio_emb: audio embeddings (AUDIO_DIM)
                - midi_feat: MIDI features (MIDI_DIM)
                - label: integer class label
        """
        self.df = dataframe.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Convert numpy arrays to tensors
        sample = {
            'song_id': row['song_id'],
            'lyrics_emb': torch.from_numpy(row['lyrics_emb']).float(),
            'lyrics_input_ids': torch.from_numpy(row['lyrics_input_ids']).long(),
            'lyrics_attention_mask': torch.from_numpy(row['lyrics_attention_mask']).long(),
            'audio_emb': torch.from_numpy(row['audio_emb']).float(),
            'midi_feat': torch.from_numpy(row['midi_feat']).float(),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'has_lyrics': row['has_lyrics'],
            'has_audio': row['has_audio'],
            'has_midi': row['has_midi']
        }
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles batching of multimodal data
    """
    # Stack all tensors
    lyrics_emb = torch.stack([item['lyrics_emb'] for item in batch])
    lyrics_input_ids = torch.stack([item['lyrics_input_ids'] for item in batch])
    lyrics_attention_mask = torch.stack([item['lyrics_attention_mask'] for item in batch])
    audio_emb = torch.stack([item['audio_emb'] for item in batch])
    midi_feat = torch.stack([item['midi_feat'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Collect metadata
    song_ids = [item['song_id'] for item in batch]
    has_lyrics = [item['has_lyrics'] for item in batch]
    has_audio = [item['has_audio'] for item in batch]
    has_midi = [item['has_midi'] for item in batch]
    
    return {
        'song_id': song_ids,
        'lyrics_emb': lyrics_emb,
        'lyrics_input_ids': lyrics_input_ids,
        'lyrics_attention_mask': lyrics_attention_mask,
        'audio_emb': audio_emb,
        'midi_feat': midi_feat,
        'label': labels,
        'has_lyrics': has_lyrics,
        'has_audio': has_audio,
        'has_midi': has_midi
    }