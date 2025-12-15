# fusion_model2.py
import torch
import torch.nn as nn
import math
from config import AUDIO_DIM, LYRICS_DIM, MIDI_DIM, FUSION_HIDDEN, NUM_HEADS, DROPOUT


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim * 3),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
    
    def forward(self, audio, lyrics, midi):
        # audio, lyrics, midi: (B, D)
        combined = torch.cat([audio, lyrics, midi], dim=-1)  # (B, 3D)
        gate = self.gate(combined)  # (B, 3D)
        gated = combined * gate  # Element-wise gating
        fused = self.transform(gated)  # (B, D)
        return fused


class ModalityEncoder(nn.Module):
    """Deep encoder for each modality with residual connections"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(3)
        ])
        
        # Residual projections
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
    
    def forward(self, x):
        residual = self.residual_proj(x)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == len(self.layers) - 1:
                out = out + residual  # Residual connection
        return out


class CrossModalAttention(nn.Module):
    """Cross-modal attention between modalities"""
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query, key_value):
        # query: (B, 1, D) or (B, T, D)
        # key_value: (B, T, D)
        attn_out, _ = self.mha(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out


class TransformerFusionBlock(nn.Module):
    """Transformer block for fused representations"""
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out


class HierarchicalClassifier(nn.Module):
    """Hierarchical classifier with multiple stages"""
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification head
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Auxiliary classifier for intermediate supervision (optional)
        self.aux_classifier = nn.Linear(hidden_dims[0], num_classes) if len(hidden_dims) > 0 else None
    
    def forward(self, x, return_aux=False):
        features = x
        aux_logits = None
        
        for i, layer in enumerate(self.feature_extractor):
            features = layer(features)
            # Capture intermediate features for auxiliary loss
            if return_aux and i == 3 and self.aux_classifier is not None:  # After first block
                aux_logits = self.aux_classifier(features)
        
        logits = self.classifier(features)
        
        if return_aux and aux_logits is not None:
            return logits, aux_logits
        return logits


class AdvancedMultimodalFusion(nn.Module):
    """
    Advanced fusion model with:
    - Deep modality-specific encoders
    - Cross-modal attention
    - Transformer-based fusion
    - Hierarchical classification
    """
    def __init__(self, num_classes, hidden_dim=FUSION_HIDDEN, num_heads=NUM_HEADS, 
                 dropout=DROPOUT, num_transformer_layers=2):
        super().__init__()
        
        H = hidden_dim
        
        # Step 1: Deep modality-specific encoders
        self.audio_encoder = ModalityEncoder(AUDIO_DIM, H, dropout)
        self.lyrics_encoder = ModalityEncoder(LYRICS_DIM, H, dropout)
        self.midi_encoder = ModalityEncoder(MIDI_DIM, H, dropout)
        
        # Step 2: Cross-modal attention layers
        self.audio_to_lyrics = CrossModalAttention(H, num_heads, dropout)
        self.audio_to_midi = CrossModalAttention(H, num_heads, dropout)
        self.lyrics_to_midi = CrossModalAttention(H, num_heads, dropout)
        
        # Step 3: Gated fusion
        self.gated_fusion = GatedFusion(H)
        
        # Step 4: Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(H)
        
        # Step 5: Transformer fusion blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerFusionBlock(H, num_heads, dropout) 
            for _ in range(num_transformer_layers)
        ])
        
        # Step 6: Modal tokens for better representation
        self.modal_tokens = nn.Parameter(torch.randn(1, 3, H))  # 3 modalities
        
        # Step 7: Hierarchical classifier
        self.classifier = HierarchicalClassifier(
            input_dim=H * 4,  # 3 modalities + 1 fused
            num_classes=num_classes,
            hidden_dims=[H * 2, H, H // 2],
            dropout=dropout
        )
        
        # Layer for combining modal representations
        self.modal_combiner = nn.Sequential(
            nn.Linear(H * 3, H),
            nn.LayerNorm(H),
            nn.GELU()
        )
    
    def forward(self, audio_emb, lyrics_emb, midi_feat, return_aux=False):
        B = audio_emb.size(0)
        
        # Step 1: Encode each modality deeply
        audio_enc = self.audio_encoder(audio_emb)  # (B, H)
        lyrics_enc = self.lyrics_encoder(lyrics_emb)  # (B, H)
        midi_enc = self.midi_encoder(midi_feat)  # (B, H)
        
        # Convert to sequences for cross-attention
        audio_seq = audio_enc.unsqueeze(1)  # (B, 1, H)
        lyrics_seq = lyrics_enc.unsqueeze(1)  # (B, 1, H)
        midi_seq = midi_enc.unsqueeze(1)  # (B, 1, H)
        
        # Step 2: Cross-modal attention
        audio_att_lyrics = self.audio_to_lyrics(audio_seq, lyrics_seq).squeeze(1)  # (B, H)
        audio_att_midi = self.audio_to_midi(audio_seq, midi_seq).squeeze(1)  # (B, H)
        lyrics_att_midi = self.lyrics_to_midi(lyrics_seq, midi_seq).squeeze(1)  # (B, H)
        
        # Enhanced representations with cross-modal info
        audio_enhanced = audio_enc + audio_att_lyrics + audio_att_midi
        lyrics_enhanced = lyrics_enc + lyrics_att_midi
        midi_enhanced = midi_enc
        
        # Step 3: Gated fusion
        fused = self.gated_fusion(audio_enhanced, lyrics_enhanced, midi_enhanced)  # (B, H)
        
        # Step 4: Create sequence for transformer
        # Combine: [modal_tokens, audio, lyrics, midi, fused]
        modal_tokens = self.modal_tokens.expand(B, -1, -1)  # (B, 3, H)
        
        sequence = torch.stack([
            audio_enhanced, 
            lyrics_enhanced, 
            midi_enhanced, 
            fused
        ], dim=1)  # (B, 4, H)
        
        sequence = torch.cat([modal_tokens, sequence], dim=1)  # (B, 7, H)
        
        # Add positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Step 5: Transformer processing
        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence)
        
        # Step 6: Extract representations
        # Use modal tokens + original enhanced features
        modal_rep = sequence[:, :3, :].mean(dim=1)  # Average of modal tokens (B, H)
        audio_final = sequence[:, 3, :]  # (B, H)
        lyrics_final = sequence[:, 4, :]  # (B, H)
        midi_final = sequence[:, 5, :]  # (B, H)
        fused_final = sequence[:, 6, :]  # (B, H)
        
        # Combine all representations
        all_features = torch.cat([
            audio_final,
            lyrics_final, 
            midi_final,
            fused_final
        ], dim=-1)  # (B, 4H)
        
        # Step 7: Hierarchical classification
        logits = self.classifier(all_features, return_aux=return_aux)
        
        return logits


# Alternative: Simpler but effective model
class ImprovedCrossAttentionFusion(nn.Module):
    """
    Improved version of original model with:
    - Deeper projections
    - Better fusion strategy
    - Hierarchical classifier
    """
    def __init__(self, num_classes, hidden_dim=FUSION_HIDDEN, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        H = hidden_dim
        
        # Deeper projections
        self.audio_proj = ModalityEncoder(AUDIO_DIM, H, dropout)
        self.lyrics_proj = ModalityEncoder(LYRICS_DIM, H, dropout)
        self.midi_proj = ModalityEncoder(MIDI_DIM, H, dropout)
        
        # Cross-modal attention
        self.mha_lyrics = nn.MultiheadAttention(H, num_heads, dropout=dropout, batch_first=True)
        self.mha_midi = nn.MultiheadAttention(H, num_heads, dropout=dropout, batch_first=True)
        
        # Gated fusion instead of simple concat
        self.gated_fusion = GatedFusion(H)
        
        # Hierarchical classifier
        self.classifier = HierarchicalClassifier(
            input_dim=H,
            num_classes=num_classes,
            hidden_dims=[H * 2, H, H // 2],
            dropout=dropout
        )
    
    def forward(self, audio_emb, lyrics_emb, midi_feat):
        # Deep encoding
        audio_enc = self.audio_proj(audio_emb)
        lyrics_enc = self.lyrics_proj(lyrics_emb)
        midi_enc = self.midi_proj(midi_feat)
        
        # Cross-attention
        audio_seq = audio_enc.unsqueeze(1)
        lyrics_seq = lyrics_enc.unsqueeze(1)
        midi_seq = midi_enc.unsqueeze(1)
        
        attn_lyrics, _ = self.mha_lyrics(audio_seq, lyrics_seq, lyrics_seq)
        attn_midi, _ = self.mha_midi(audio_seq, midi_seq, midi_seq)
        
        audio_enhanced = audio_enc + attn_lyrics.squeeze(1) + attn_midi.squeeze(1)
        
        # Gated fusion
        fused = self.gated_fusion(audio_enhanced, lyrics_enc, midi_enc)
        
        # Classification
        logits = self.classifier(fused)
        return logits