# train2.py - Training with built-in evaluation and visualization
import os
import sys
import pickle
import random
import platform
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from transformers import BertModel, get_cosine_schedule_with_warmup

from config import (
    CACHE_PATH, BATCH_SIZE, EPOCHS, LR_HEAD, LR_BACKBONE, 
    DEFAULT_NUM_WORKERS, PATIENCE, WARMUP_RATIO, SEED
)
from fusion_model2 import AdvancedMultimodalFusion, ImprovedCrossAttentionFusion
from dataset import MultimodalDataset, collate_fn

# Base results folder. Semua keluaran akan disimpan di sini.
BASE_RESULTS = 'results2'

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class CombinedLoss(nn.Module):
    """Combined loss: Focal + Label Smoothing + Auxiliary"""
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1, aux_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smooth = LabelSmoothingLoss(num_classes, smoothing)
        self.aux_weight = aux_weight
    
    def forward(self, logits, targets, aux_logits=None):
        focal_loss = self.focal(logits, targets)
        smooth_loss = self.label_smooth(logits, targets)
        main_loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        if aux_logits is not None:
            aux_loss = self.focal(aux_logits, targets)
            total_loss = main_loss + self.aux_weight * aux_loss
            return total_loss, main_loss, aux_loss
        
        return main_loss, main_loss, torch.tensor(0.0)


def attention_pooling(last_hidden, attn_mask):
    """Attention-based pooling for BERT outputs"""
    weights = last_hidden.mean(dim=2)
    weights = weights.masked_fill(attn_mask == 0, -1e9)
    weights = torch.softmax(weights, dim=1).unsqueeze(-1)
    pooled = (last_hidden * weights).sum(dim=1)
    return pooled


def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping with model checkpointing"""
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return True
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return True
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return False


def evaluate(model, bert_model, loader, device, criterion, use_aux=False):
    """Evaluate model on validation/test set"""
    model.eval()
    if bert_model is not None:
        bert_model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            audio = batch['audio_emb'].to(device)
            midi = batch['midi_feat'].to(device)
            labels = batch['label'].to(device)
            lyrics_emb = batch['lyrics_emb'].to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                if use_aux:
                    logits, aux_logits = model(audio, lyrics_emb, midi, return_aux=True)
                    loss, _, _ = criterion(logits, labels, aux_logits)
                else:
                    logits = model(audio, lyrics_emb, midi)
                    loss, _, _ = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], []
    
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return total_loss / len(loader.dataset), acc, p, r, f1, all_preds, all_labels


def plot_training_curves(all_histories, model_name, save_dir=None):
    """Plot training curves for all folds"""
    if save_dir is None:
        save_dir = os.path.join(BASE_RESULTS, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Training Curves', fontsize=16, fontweight='bold')
    
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy'),
        ('train_mae', 'val_mae', 'MAE'),
        ('val_f1', None, 'Validation F1-Score')
    ]
    
    for idx, (train_metric, val_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for fold_name, hist in all_histories.items():
            epochs = range(1, len(hist[train_metric]) + 1)
            
            if train_metric:
                ax.plot(epochs, hist[train_metric], 
                       label=f'{fold_name} Train', linestyle='--', alpha=0.6)
            
            if val_metric and val_metric in hist:
                ax.plot(epochs, hist[val_metric], 
                       label=f'{fold_name} Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_dir}/{model_name}_training_curves.png")


def plot_confusion_matrices(model_name, label_names, save_dir=None):
    """Plot confusion matrices for all folds"""
    if save_dir is None:
        save_dir = os.path.join(BASE_RESULTS, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    import glob
    pattern = os.path.join(BASE_RESULTS, f'fold_*_{model_name}', 'confusion.npy')
    cm_files = sorted(glob.glob(pattern))
    
    if not cm_files:
        print(f"⚠ No confusion matrices found for {model_name}")
        return
    
    n_folds = len(cm_files)
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    
    if n_folds == 1:
        axes = [axes]
    
    for idx, cm_file in enumerate(cm_files):
        cm = np.load(cm_file)
        fold_name = os.path.basename(os.path.dirname(cm_file))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names,
                   ax=axes[idx], cbar_kws={'label': 'Proportion'})
        axes[idx].set_title(f'{fold_name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrices saved to {save_dir}/{model_name}_confusion_matrices.png")


def plot_performance_summary(fold_summaries, model_name, save_dir=None):
    """Plot performance summary across folds"""
    if save_dir is None:
        save_dir = os.path.join(BASE_RESULTS, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(fold_summaries)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=16, fontweight='bold')
    
    # Bar plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i*width, df[metric], width, label=metric.capitalize())
    
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance per Fold')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels([f"Fold {i}" for i in df['fold']])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    data_to_plot = [df[m].values for m in metrics]
    axes[1].boxplot(data_to_plot, labels=[m.capitalize() for m in metrics])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance summary saved to {save_dir}/{model_name}_performance_summary.png")


def run_training(model_type='advanced'):
    """
    Main training function with built-in evaluation
    
    Args:
        model_type: 'advanced' for AdvancedMultimodalFusion
                   'improved' for ImprovedCrossAttentionFusion
    """
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print(f"MULTIMODAL MUSIC CLASSIFICATION - MODEL: {model_type.upper()}")
    print("="*80)
    print(f"Device: {device}")
    
    # Load cached dataset
    if not os.path.exists(CACHE_PATH):
        print(f"❌ Cache missing: {CACHE_PATH}")
        print("Please run: python prepare_dataset.py")
        sys.exit(1)
    
    with open(CACHE_PATH, 'rb') as fh:
        obj = pickle.load(fh)
    
    df = obj['df']
    label_encoder = obj.get('label_encoder', None)
    num_classes = len(label_encoder.classes_) if label_encoder else int(df['label'].nunique())
    
    print(f"\n📊 Dataset Info:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Classes: {list(label_encoder.classes_)}")
    print(f"  - Class distribution:")
    for cls, count in df['label'].value_counts().sort_index().items():
        cls_name = label_encoder.inverse_transform([cls])[0]
        print(f"      {cls_name}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Prepare cross-validation
    X = df.index.values
    y = df['label'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Set number of workers
    num_workers = DEFAULT_NUM_WORKERS
    if platform.system() == "Windows":
        num_workers = 0
    
    # Prepare BERT model
    print(f"\n🔧 Loading BERT model...")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(device)
    for p in bert_model.parameters():
        p.requires_grad = False
    
    fold_summaries = []
    all_histories = {}
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print("\n" + "="*80)
        print(f"FOLD {fold} / 5")
        print("="*80)
        
        # Prepare datasets
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        train_ds = MultimodalDataset(train_df)
        val_ds = MultimodalDataset(val_df)
        
        # Weighted sampling for imbalanced classes
        train_labels = train_df['label'].values
        class_counts = np.bincount(train_labels)
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Data loaders
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=num_workers, pin_memory=(device.type=='cuda'), 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type=='cuda'), 
            collate_fn=collate_fn
        )
        
        # Initialize model
        if model_type == 'advanced':
            model = AdvancedMultimodalFusion(
                num_classes=num_classes,
                num_transformer_layers=2
            ).to(device)
            use_aux = True
        else:  # improved
            model = ImprovedCrossAttentionFusion(
                num_classes=num_classes
            ).to(device)
            use_aux = False
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if p.requires_grad], 
             'lr': LR_HEAD},
            {'params': list(bert_model.parameters()), 'lr': LR_BACKBONE}
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = max(1, len(train_loader) * EPOCHS)
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = CombinedLoss(
            num_classes=num_classes,
            alpha=torch.FloatTensor(class_weights).to(device),
            gamma=2.0,
            smoothing=0.1,
            aux_weight=0.3
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Early stopping
        early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'train_mae': [], 'val_mae': [], 
            'val_f1': [], 'val_precision': [], 'val_recall': []
        }
        
        best_f1 = -1.0
        fine_tune_bert_after = 3
        
        # Training loop
        for epoch in range(1, EPOCHS + 1):
            # Enable BERT fine-tuning after certain epochs
            if epoch >= fine_tune_bert_after:
                bert_model.train()
                for p in bert_model.parameters():
                    p.requires_grad = True
            else:
                bert_model.eval()
                for p in bert_model.parameters():
                    p.requires_grad = False
            
            model.train()
            train_loss = 0.0
            preds_train = []
            trues_train = []
            
            # Training batches
            for batch_idx, batch in enumerate(train_loader):
                audio = batch['audio_emb'].to(device)
                midi = batch['midi_feat'].to(device)
                labels = batch['label'].to(device)
                
                # Use BERT if fine-tuning
                if epoch >= fine_tune_bert_after and 'lyrics_input_ids' in batch:
                    input_ids = batch['lyrics_input_ids'].to(device)
                    attn = batch['lyrics_attention_mask'].to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                        bert_out = bert_model(input_ids=input_ids, attention_mask=attn)
                        last = bert_out.last_hidden_state
                        lyrics_emb = attention_pooling(last, attn)
                else:
                    lyrics_emb = batch['lyrics_emb'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    if use_aux:
                        logits, aux_logits = model(audio, lyrics_emb, midi, return_aux=True)
                        loss, main_loss, aux_loss = criterion(logits, labels, aux_logits)
                    else:
                        logits = model(audio, lyrics_emb, midi)
                        loss, main_loss, aux_loss = criterion(logits, labels)
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                # Accumulate metrics
                train_loss += loss.item() * labels.size(0)
                preds_train.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                trues_train.extend(labels.cpu().numpy().tolist())
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = accuracy_score(trues_train, preds_train) if len(trues_train) > 0 else 0.0
            train_mae = np.mean(np.abs(np.array(preds_train) - np.array(trues_train))) if len(trues_train) > 0 else 0.0
            
            # Validation
            val_loss, val_acc, val_p, val_r, val_f1, preds_val, trues_val = evaluate(
                model, bert_model, val_loader, device, criterion, use_aux=use_aux
            )
            val_mae = np.mean(np.abs(np.array(preds_val) - np.array(trues_val))) if len(trues_val) > 0 else 0.0
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_p)
            history['val_recall'].append(val_r)
            
            print(f'Epoch {epoch:02d}/{EPOCHS} | '
                  f'TrLoss: {train_loss:.4f} TrAcc: {train_acc:.4f} | '
                  f'VlLoss: {val_loss:.4f} VlAcc: {val_acc:.4f} '
                  f'VlF1: {val_f1:.4f}')
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                os.makedirs(BASE_RESULTS, exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'bert': bert_model.state_dict(),
                    'label_encoder': label_encoder,
                    'model_type': model_type,
                    'epoch': epoch,
                    'best_f1': best_f1
                }, os.path.join(BASE_RESULTS, f'best_fold{fold}_{model_type}.pt'))
                print(f'  ✓ New best model saved (F1: {best_f1:.4f})')
            
            # Early stopping check
            if early_stopping(val_f1):
                if early_stopping.early_stop:
                    print(f'⚠ Early stopping at epoch {epoch}')
                    break
        
        # Save fold results
        fold_dir = os.path.join(BASE_RESULTS, f'fold_{fold}_{model_type}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save history
        with open(os.path.join(fold_dir, 'history.pkl'), 'wb') as fh:
            pickle.dump(history, fh)
        
        # Save confusion matrix and classification report
        if len(trues_val) > 0:
            cm = confusion_matrix(trues_val, preds_val)
            np.save(os.path.join(fold_dir, 'confusion.npy'), cm)
            
            report = classification_report(
                trues_val, preds_val, 
                target_names=label_encoder.classes_, 
                zero_division=0
            )
            with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as rr:
                rr.write(report)
            
            print("\n📋 Classification Report:")
            print(report)
        
        # Store fold summary
        fold_summaries.append({
            'fold': fold,
            'accuracy': history['val_acc'][-1] if history['val_acc'] else 0.0,
            'precision': history['val_precision'][-1] if history['val_precision'] else 0.0,
            'recall': history['val_recall'][-1] if history['val_recall'] else 0.0,
            'f1': history['val_f1'][-1] if history['val_f1'] else 0.0,
            'best_f1': best_f1
        })
        all_histories[f'fold_{fold}'] = history
    
    # Save overall summary
    overall_dir = os.path.join(BASE_RESULTS, f'overall_{model_type}')
    os.makedirs(overall_dir, exist_ok=True)
    with open(os.path.join(overall_dir, 'summary.pkl'), 'wb') as fh:
        pickle.dump({
            'folds': fold_summaries, 
            'histories': all_histories,
            'model_type': model_type
        }, fh)
    
    # Print final summary
    print("\n" + "="*80)
    print(f"FINAL SUMMARY - {model_type.upper()} MODEL")
    print("="*80)
    print(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Best F1':<10}")
    print("-"*80)
    
    for f in fold_summaries:
        print(f"{f['fold']:<6} {f['accuracy']:<10.4f} {f['precision']:<10.4f} "
              f"{f['recall']:<10.4f} {f['f1']:<10.4f} {f['best_f1']:<10.4f}")
    
    accs = [f['accuracy'] for f in fold_summaries]
    precs = [f['precision'] for f in fold_summaries]
    recs = [f['recall'] for f in fold_summaries]
    f1s = [f['f1'] for f in fold_summaries]
    best_f1s = [f['best_f1'] for f in fold_summaries]
    
    print("-"*80)
    print(f"\n📈 CROSS-VALIDATION RESULTS:")
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1-Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Best F1:   {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_training_curves(all_histories, model_type)
    plot_confusion_matrices(model_type, label_encoder.classes_)
    plot_performance_summary(fold_summaries, model_type)
    
    print(f"\n✅ Training finished! All results saved under {BASE_RESULTS}/")
    print(f"📂 Model checkpoints: {BASE_RESULTS}/best_fold*_{model_type}.pt")
    print(f"📊 Visualizations: {BASE_RESULTS}/plots/")
    print(f"📄 Reports: {BASE_RESULTS}/fold_*_{model_type}/")


if __name__ == "__main__":
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser(description='Train multimodal music classifier')
    parser.add_argument('--model', type=str, default='advanced', 
                       choices=['advanced', 'improved'],
                       help='Model type: advanced (slower, better) or improved (faster)')
    args = parser.parse_args()
    
    run_training(model_type=args.model)
