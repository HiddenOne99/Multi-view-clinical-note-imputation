# ======================== USER SETTINGS =========================
TRAIN_JSON = "path_to_the_json_file"
VAL_JSON = "path_to_the_json_file"
TEST_JSON = "path_to_the_json_file"
OUT_DIR = "output"
MODEL_NAME = "your_model_name_from_huggingface_or_local"
CACHE_DIR = "your_cache_directory_for_the_model"
GPU_LIST = "0,1,2,3"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 20
LR_ENCODER = 2e-5
LR_NEW = 2e-5
LR_ACT = 5e-3
WEIGHT_DECAY = 0.0001
SEED = 42
NUM_WORKERS = 4
BF16 = True
GRAD_ACCUM_STEPS = 8
DROPOUT = 0.1
STEPS_PER_LOG = 1
THRESHOLD = 0.5
FREEZE_EMBEDDINGS = False

# ----------- Pipeline-specific hyperparameters -----------
D_HIDDEN = 768
D_SEC_ID = 64

NUM_VIEWS = 2
NUM_ATTN_HEADS = 2
EDGE_ID_DIM = 64
ROUTER_CTX_DIM = 64

ACT_MASK_DROPOUT = 0.2
ACT_HIDDEN_DIM = 256

LAMBDA_ACT = 1.0
LAMBDA_TASK = 1.0
LAMBDA_MSG_DECORR = 1.0

MSG_DECORR_MARGIN = 0.1

MASK_NEG_INF = -1e9

USE_LR_SCHEDULER = False
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.1

CHECKPOINT_MODE = "val_mean"
VAL_STEPS = 0
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_LIST

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score

SECTION_NAMES = [
    "past medical history",
    "chief complaint",
    "family history",
    "physical exam",
    "allergies",
    "social history",
    "medications on admission",
    "present illness"
]
K_SECTIONS = len(SECTION_NAMES)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_cache_dir(cache_dir: str):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    test_file = cache_path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise RuntimeError(f"Cache directory {cache_dir} is not writable: {e}")


def load_json_data(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def tokenize_and_save_split(
    data: List[Dict],
    tokenizer,
    split_name: str,
    out_dir: Path,
    max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    N = len(data)
    tokenized_sections = torch.zeros(N, K_SECTIONS, 3, max_length, dtype=torch.long)
    exist_mask = torch.zeros(N, K_SECTIONS, dtype=torch.float32)

    for idx, note in enumerate(data):
        sections = note['sections']
        for sec_idx, sec_name in enumerate(SECTION_NAMES):
            sec_text = sections[sec_name]
            if sec_text.strip():
                exist_mask[idx, sec_idx] = 1.0
                encoded = tokenizer(
                    sec_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                tokenized_sections[idx, sec_idx, 0, :] = encoded['input_ids'].squeeze(0)
                tokenized_sections[idx, sec_idx, 1, :] = encoded['attention_mask'].squeeze(0)
                if 'token_type_ids' in encoded:
                    tokenized_sections[idx, sec_idx, 2, :] = encoded['token_type_ids'].squeeze(0)

    torch.save(tokenized_sections, out_dir / f"tokenized_sections_{split_name}.pt")
    torch.save(exist_mask, out_dir / f"exist_mask_{split_name}.pt")
    return tokenized_sections, exist_mask


class SectionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenized_sections: torch.Tensor, exist_mask: torch.Tensor):
        self.data = data
        self.tokenized_sections = tokenized_sections
        self.exist_mask = exist_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'tokenized_sections': self.tokenized_sections[idx],
            'exist_mask': self.exist_mask[idx],
            'label': torch.tensor(int(self.data[idx]['labels']), dtype=torch.float32)
        }

def build_leave_one_out_mask(mask: torch.Tensor) -> torch.Tensor:

    B, K = mask.shape
    eye = torch.eye(K, device=mask.device, dtype=mask.dtype)
    not_self = 1.0 - eye
    C = mask[:, None, :] * not_self[None, :, :]
    return C


def masked_mean_pool(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:

    counts = C.sum(dim=-1, keepdim=True).clamp(min=1.0)
    pooled = torch.bmm(C, X) / counts
    return pooled

class ActivationHead(nn.Module):

    def __init__(self, d: int, d_id: int, K: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        input_dim = d + d_id + K
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FiLMRouter(nn.Module):

    def __init__(self, d: int, R: int, ctx_dim: int = ROUTER_CTX_DIM):
        super().__init__()
        self.ctx_proj = nn.Linear(d, ctx_dim)
        self.q_to_gamma = nn.Linear(1, ctx_dim)
        self.q_to_beta = nn.Linear(1, ctx_dim)
        self.out_proj = nn.Linear(ctx_dim, R)

        nn.init.ones_(self.q_to_gamma.bias)
        nn.init.zeros_(self.q_to_gamma.weight)
        nn.init.zeros_(self.q_to_beta.bias)
        nn.init.zeros_(self.q_to_beta.weight)

    def forward(self, q_scalar: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:

        ctx_small = self.ctx_proj(ctx)
        gamma = self.q_to_gamma(q_scalar)
        beta = self.q_to_beta(q_scalar)
        modulated = gamma * ctx_small + beta
        return self.out_proj(F.relu(modulated))


class HypergraphImputer(nn.Module):

    def __init__(self, K: int, R: int, d: int, H: int = NUM_ATTN_HEADS,
                 edge_id_dim: int = EDGE_ID_DIM,
                 router_ctx_dim: int = ROUTER_CTX_DIM):
        super().__init__()
        assert d % H == 0, f"d={d} must be divisible by H={H}"
        self.K = K
        self.R = R
        self.d = d
        self.H = H
        self.d_h = d // H

        self.edge_tokens = nn.Parameter(torch.randn(R, edge_id_dim))
        nn.init.normal_(self.edge_tokens, std=0.5)

        self.shared_mlp = nn.Sequential(
            nn.Linear(d + edge_id_dim, d),
            nn.GELU(),
        )

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_Ke = nn.Linear(d, d, bias=False)
        self.W_Ve = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

        self.router = nn.ModuleList([
            FiLMRouter(d, R, router_ctx_dim)
            for _ in range(K)
        ])

    def forward(
        self,
        X: torch.Tensor,
        M_ctx: torch.Tensor,
        Q_bar: torch.Tensor,
        query_seed: torch.Tensor,
        return_messages: bool = False
    ):
        B, K, d = X.shape
        H, d_h = self.H, self.d_h

        C_imp = build_leave_one_out_mask(M_ctx)
        has_any_source = (C_imp.sum(dim=-1) > 0).float()

        pooled_ctx = masked_mean_pool(X, C_imp)

        counts = C_imp.sum(dim=-1, keepdim=True).clamp(min=1.0)
        edge_messages = []
        for r in range(self.R):
            token_r = self.edge_tokens[r].expand(B, K, -1)
            mlp_input = torch.cat([X, token_r], dim=-1)
            transformed = self.shared_mlp(mlp_input)
            msg_r = torch.bmm(C_imp, transformed) / counts
            edge_messages.append(msg_r)
        M_edge = torch.stack(edge_messages, dim=2)

        route_logits = torch.zeros(B, K, self.R, device=X.device, dtype=X.dtype)
        for s in range(K):
            route_logits[:, s, :] = self.router[s](
                Q_bar[:, s:s+1],
                pooled_ctx[:, s, :],
            )

        Q = self.W_Q(query_seed).view(B, K, H, d_h).transpose(1, 2)

        K_edge = self.W_Ke(M_edge).view(B, K, self.R, H, d_h).permute(0, 3, 1, 2, 4)
        V_edge = self.W_Ve(M_edge).view(B, K, self.R, H, d_h).permute(0, 3, 1, 2, 4)

        edge_logits = torch.einsum("bhsd,bhsrd->bhsr", Q, K_edge) / math.sqrt(d_h)
        edge_logits = edge_logits + route_logits[:, None, :, :]

        beta = F.softmax(edge_logits, dim=-1)
        beta = beta.nan_to_num(0.0)

        out = torch.einsum("bhsr,bhsrd->bhsd", beta, V_edge)
        out = out.transpose(1, 2).contiguous().view(B, K, d)
        Z_imp = self.out_proj(out)

        Z_imp = Z_imp * has_any_source.unsqueeze(-1)

        if return_messages:
            return Z_imp, M_edge
        return Z_imp


class SectionClassifier(nn.Module):

    def __init__(
        self,
        encoder_name: str,
        cache_dir: str,
        d: int = D_HIDDEN,
        d_id: int = D_SEC_ID,
        K: int = K_SECTIONS,
        R: int = NUM_VIEWS,
        H: int = NUM_ATTN_HEADS,
        dropout: float = DROPOUT,
        act_hidden_dim: int = ACT_HIDDEN_DIM,
        act_mask_dropout: float = ACT_MASK_DROPOUT,
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.d_id = d_id
        self.act_mask_dropout_rate = act_mask_dropout

        self.encoder = AutoModel.from_pretrained(encoder_name, cache_dir=cache_dir)
        d_sec = self.encoder.config.hidden_size

        self.section_id_emb = nn.Parameter(torch.zeros(K, d_id))
        nn.init.normal_(self.section_id_emb, std=0.02)

        self.section_proj_clean = nn.Sequential(
            nn.Linear(d_sec + d_id, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )
        self.section_proj_dropout = nn.Dropout(dropout)

        self.activation_head = ActivationHead(d, d_id, K, act_hidden_dim, dropout)

        self.imputer = HypergraphImputer(K, R, d, H)

        self.cls_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, 1)

    def encode_sections(
        self,
        tokenized_sections: torch.Tensor,
        exist_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B = tokenized_sections.size(0)
        device = tokenized_sections.device
        d_sec = self.encoder.config.hidden_size

        E_full = torch.zeros(B, self.K, d_sec, device=device)

        for sec_idx in range(self.K):
            sec_exists = exist_mask[:, sec_idx]
            exists_indices = (sec_exists == 1.0).nonzero(as_tuple=True)[0]

            if len(exists_indices) > 0:
                input_ids = tokenized_sections[exists_indices, sec_idx, 0, :]
                attention_mask = tokenized_sections[exists_indices, sec_idx, 1, :]
                token_type_ids = tokenized_sections[exists_indices, sec_idx, 2, :]

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                cls_embed = outputs.last_hidden_state[:, 0, :]
                E_full[exists_indices, sec_idx, :] = cls_embed

        T_expand = self.section_id_emb[None, :, :].expand(B, self.K, self.d_id)
        X_clean = self.section_proj_clean(torch.cat([E_full, T_expand], dim=-1))
        X = self.section_proj_dropout(X_clean)

        return X, X_clean, E_full

    def build_missing_query_seed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        d_sec = self.encoder.config.hidden_size

        zero_content = torch.zeros(
                batch_size,
                self.K,
                d_sec,
                device=device,
                dtype=self.section_id_emb.dtype
                )

        T_expand = self.section_id_emb[None, :, :].expand(batch_size, self.K, self.d_id)
        query_seed = self.section_proj_clean(torch.cat([zero_content, T_expand], dim=-1))
        return query_seed

    def activation_branch(
        self,
        X: torch.Tensor,
        M_orig: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, K, d = X.shape

        C_act = build_leave_one_out_mask(M_orig)

        G_act = masked_mean_pool(X, C_act)

        M_other = C_act.float()

        if training and self.act_mask_dropout_rate > 0:
            keep_prob = 1.0 - self.act_mask_dropout_rate
            keep_mask = torch.bernoulli(
                torch.full_like(M_other, keep_prob)
            )
            M_other = M_other * keep_mask

        T_expand = self.section_id_emb.detach()[None, :, :].expand(B, K, self.d_id)

        act_input = torch.cat([G_act, T_expand, M_other], dim=-1)

        Q_logits = self.activation_head(act_input)

        Q_bar = torch.sigmoid(Q_logits).detach()

        return Q_logits, Q_bar

    def fuse_and_classify(
        self,
        X: torch.Tensor,
        Z_imp: torch.Tensor,
        Q_bar: torch.Tensor,
        M_orig: torch.Tensor,
    ) -> torch.Tensor:

        trust = Q_bar * M_orig
        t = trust.unsqueeze(-1)

        Z_final = t * X + (1.0 - t) * Z_imp

        h = Z_final.mean(dim=1)

        h = self.cls_dropout(h)
        logits = self.classifier(h)
        return logits

    def forward(
        self,
        tokenized_sections: torch.Tensor,
        exist_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_losses: bool = False
    ):

        B = tokenized_sections.size(0)
        M_orig = exist_mask
        is_training = self.training

        X, X_clean, E_full = self.encode_sections(tokenized_sections, M_orig)

        query_seed = self.build_missing_query_seed(B, tokenized_sections.device)

        Q_logits, Q_bar = self.activation_branch(X.detach(), M_orig, training=is_training)

        if return_losses:

            Z_imp_task, M_edge_task = self.imputer(X_clean, M_orig, Q_bar, query_seed, return_messages=True)
            logits = self.fuse_and_classify(X_clean, Z_imp_task, Q_bar, M_orig)

            return {
                'logits': logits,
                'Q_logits': Q_logits,
                'M_edge_task': M_edge_task,
            }
        else:
            Z_imp_task = self.imputer(X_clean, M_orig, Q_bar, query_seed)
            logits = self.fuse_and_classify(X_clean, Z_imp_task, Q_bar, M_orig)
            return logits

def compute_activation_loss(Q_logits: torch.Tensor, M_orig: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(Q_logits, M_orig, reduction='mean')


def compute_message_decorrelation_loss(
    M_edge: torch.Tensor,
    margin: float = MSG_DECORR_MARGIN
) -> torch.Tensor:

    B, K, R, d = M_edge.shape

    if R < 2:
        return M_edge.new_tensor(0.0)

    flat = M_edge.reshape(B * K, R, d)
    normed = F.normalize(flat, p=2, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))

    eye = torch.eye(R, device=M_edge.device, dtype=M_edge.dtype)
    off_diag = sim * (1.0 - eye)[None, :, :]
    hinge = torch.clamp(off_diag - margin, min=0.0)

    num_pairs = B * K * R * (R - 1)
    return hinge.sum() / max(num_pairs, 1)

def compute_class_weight(data: List[Dict]) -> float:
    labels = [int(item['labels']) for item in data]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    if num_pos == 0:
        return 1.0
    return num_neg / num_pos


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task_criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    steps_per_log: int,
    epoch: int,
    rank: int,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    val_state: Optional[Dict] = None,
) -> Dict[str, float]:

    model.train()
    total_task_loss = 0.0
    total_act_loss = 0.0
    total_msg_decorr_loss = 0.0
    total_combined_loss = 0.0
    num_batches = 0

    raw_model = model.module if hasattr(model, 'module') else model

    do_within_epoch_val = (val_state is not None and val_state.get('val_steps', 0) > 0)

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        tokenized_sections = batch['tokenized_sections'].to(device)
        exist_mask = batch['exist_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if BF16 else torch.float16):
                out = model(tokenized_sections, exist_mask, labels=labels, return_losses=True)

                L_task = task_criterion(out['logits'], labels)

                L_act = compute_activation_loss(out['Q_logits'], exist_mask)

                L_msg_decorr = compute_message_decorrelation_loss(out['M_edge_task'])

                loss = (LAMBDA_TASK * L_task + LAMBDA_ACT * L_act
                     + LAMBDA_MSG_DECORR * L_msg_decorr)
        else:
            out = model(tokenized_sections, exist_mask, labels=labels, return_losses=True)
            L_task = task_criterion(out['logits'], labels)
            L_act = compute_activation_loss(out['Q_logits'], exist_mask)
            L_msg_decorr = compute_message_decorrelation_loss(out['M_edge_task'])
            loss = (LAMBDA_TASK * L_task + LAMBDA_ACT * L_act
                    + LAMBDA_MSG_DECORR * L_msg_decorr)

        scaled_loss = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_task_loss += L_task.item()
        total_act_loss += L_act.item()
        total_msg_decorr_loss += L_msg_decorr.item()
        total_combined_loss += loss.item()
        num_batches += 1

        if rank == 0 and (step + 1) % steps_per_log == 0:
            print(
                f"Epoch {epoch} Step {step+1}/{len(dataloader)} "
                f"Loss: {loss.item():.4f} "
                f"(task={L_task.item():.4f}, act={L_act.item():.4f}, "
                f"msg_decorr={L_msg_decorr.item():.4f})"
            )

        if do_within_epoch_val and (step + 1) % val_state['val_steps'] == 0:
            vs = val_state
            mid_val_score = 0.0

            if rank == 0:
                eval_model = vs['eval_model_fn']()
                mid_metrics = evaluate(eval_model, vs['val_loader'],
                                       vs['val_criterion'], device, vs['threshold'])
                mid_val_score = model_selection_score(mid_metrics, CHECKPOINT_MODE)

                print(f"  [Mid-epoch val @ step {step+1}] "
                      f"ROC={mid_metrics['roc_auc']:.4f} "
                      f"mF1={mid_metrics['micro_f1']:.4f} "
                      f"MF1={mid_metrics['macro_f1']:.4f} "
                      f"score={mid_val_score:.4f}")

                if mid_val_score > vs['best_val_score']:
                    vs['best_val_score'] = float(mid_val_score)
                    vs['best_val_metrics'] = {
                        'loss': float(mid_metrics['loss']),
                        'roc_auc': float(mid_metrics['roc_auc']),
                        'micro_f1': float(mid_metrics['micro_f1']),
                        'macro_f1': float(mid_metrics['macro_f1'])
                    }
                    torch.save({'model': eval_model.state_dict()}, vs['best_path'])
                    print(f"  [INFO] Saved mid-epoch checkpoint ({CHECKPOINT_MODE}={mid_val_score:.4f})")

            if vs.get('scheduler') is not None:
                if vs['world_size'] > 1:
                    score_tensor = torch.tensor(
                        [mid_val_score if rank == 0 else 0.0], device=device)
                    dist.broadcast(score_tensor, src=0)
                    vs['scheduler'].step(score_tensor.item())
                else:
                    vs['scheduler'].step(mid_val_score)

            model.train()

    if num_batches % grad_accum_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return {
        'combined': total_combined_loss / max(num_batches, 1),
        'task': total_task_loss / max(num_batches, 1),
        'act': total_act_loss / max(num_batches, 1),
        'msg_decorr': total_msg_decorr_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Dict:

    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        tokenized_sections = batch['tokenized_sections'].to(device)
        exist_mask = batch['exist_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        logits = model(tokenized_sections, exist_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).squeeze(1).numpy()
    all_labels = torch.cat(all_labels, dim=0).squeeze(1).numpy()

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)

    roc_auc = roc_auc_score(all_labels, probs)
    micro_f1 = f1_score(all_labels, preds, average='micro')
    macro_f1 = f1_score(all_labels, preds, average='macro')
    avg_loss = total_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'roc_auc': roc_auc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }


def model_selection_score(metrics: Dict[str, float], mode: str = CHECKPOINT_MODE) -> float:

    if mode == "val_mean":
        vals = [
            float(metrics["roc_auc"]),
            float(metrics["micro_f1"]),
            float(metrics["macro_f1"]),
        ]
        if any(np.isnan(v) for v in vals):
            return float("-inf")
        return float(sum(vals) / 3.0)
    elif mode == "val_roc":
        v = float(metrics["roc_auc"])
        return float("-inf") if np.isnan(v) else v
    elif mode == "val_loss":
        v = float(metrics["loss"])
        return float("-inf") if np.isnan(v) else -v
    else:
        raise ValueError(f"Unknown checkpoint mode: {mode}. Use 'val_mean', 'val_roc', or 'val_loss'.")

def train_worker(rank: int, world_size: int, args):

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    out_dir = args['out_dir']

    if rank == 0:
        print("Loading data from disk...")

    train_data = load_json_data(args['train_json'])
    val_data = load_json_data(args['val_json'])
    test_data = load_json_data(args['test_json'])

    tokenized_train = torch.load(out_dir / "tokenized_sections_train.pt")
    tokenized_val = torch.load(out_dir / "tokenized_sections_val.pt")
    tokenized_test = torch.load(out_dir / "tokenized_sections_test.pt")

    exist_mask_train = torch.load(out_dir / "exist_mask_train.pt")
    exist_mask_val = torch.load(out_dir / "exist_mask_val.pt")
    exist_mask_test = torch.load(out_dir / "exist_mask_test.pt")

    pos_weight = args['pos_weight']

    train_dataset = SectionDataset(train_data, tokenized_train, exist_mask_train)
    val_dataset = SectionDataset(val_data, tokenized_val, exist_mask_val)
    test_dataset = SectionDataset(test_data, tokenized_test, exist_mask_test)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SectionClassifier(MODEL_NAME, CACHE_DIR)
    model = model.to(device)

    if FREEZE_EMBEDDINGS:
        emb_mod = None
        if hasattr(model.encoder, "embeddings"):
            emb_mod = model.encoder.embeddings
        elif hasattr(model.encoder, "bert") and hasattr(model.encoder.bert, "embeddings"):
            emb_mod = model.encoder.bert.embeddings
        if emb_mod is None:
            raise ValueError("Could not find embedding module on the backbone encoder.")
        for p in emb_mod.parameters():
            p.requires_grad = False
        if rank == 0:
            frozen = sum(p.numel() for p in emb_mod.parameters())
            print(f"Froze embedding layer parameters: {frozen}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=True
        )

    task_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    raw = model.module if hasattr(model, 'module') else model
    encoder_param_ids = set(id(p) for p in raw.encoder.parameters())
    act_head_param_ids = set(id(p) for p in raw.activation_head.parameters())

    encoder_group = []
    act_group = []
    pipeline_group = []
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_group.append(param)
        elif id(param) in act_head_param_ids:
            act_group.append(param)
        else:
            pipeline_group.append(param)

    optimizer = torch.optim.AdamW([
        {'params': encoder_group,  'lr': LR_ENCODER},
        {'params': act_group,      'lr': LR_ACT},
        {'params': pipeline_group, 'lr': LR_NEW},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR)

    if rank == 0:
        n_enc = sum(p.numel() for p in encoder_group)
        n_act = sum(p.numel() for p in act_group)
        n_pipe = sum(p.numel() for p in pipeline_group)
        print(f"Optimizer param groups:")
        print(f"  encoder:  {n_enc:,} params @ lr={LR_ENCODER}")
        print(f"  act_head: {n_act:,} params @ lr={LR_ACT}")
        print(f"  pipeline: {n_pipe:,} params @ lr={LR_NEW}")
        if scheduler is not None:
            print(f"  LR scheduler: ReduceLROnPlateau(patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
            print(f"  Scheduler tracks: {CHECKPOINT_MODE}")
        else:
            print(f"  LR scheduler: None (constant LR)")

    use_amp = BF16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not BF16)

    training_logs = []
    best_val_score = float("-inf")
    best_val_metrics = None
    best_path = out_dir / "best_model.pt"
    val_score = 0.0

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Starting training for {EPOCHS} epochs")
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        print(f"Batch: {BATCH_SIZE}, LR_enc: {LR_ENCODER}, LR_act: {LR_ACT}, LR_new: {LR_NEW}, pos_weight: {pos_weight:.4f}")
        print(f"d_hidden: {D_HIDDEN}, d_sec_id: {D_SEC_ID}, R: {NUM_VIEWS}, H: {NUM_ATTN_HEADS}")
        print(f"lambda_act: {LAMBDA_ACT}, lambda_task: {LAMBDA_TASK}")
        print(f"lambda_msg_decorr: {LAMBDA_MSG_DECORR}, msg_decorr_margin: {MSG_DECORR_MARGIN}")
        print(f"act_mask_dropout: {ACT_MASK_DROPOUT}")
        print(f"Mixed precision (BF16): {use_amp}")
        print(f"Checkpoint selection: {CHECKPOINT_MODE}")
        if VAL_STEPS > 0:
            print(f"Within-epoch validation: every {VAL_STEPS} steps")
        else:
            print(f"Within-epoch validation: disabled (end-of-epoch only)")
        print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        val_state = None
        if VAL_STEPS > 0:
            val_state = {
                'val_loader': val_loader,
                'eval_model_fn': lambda: model.module if world_size > 1 else model,
                'val_criterion': task_criterion,
                'threshold': THRESHOLD,
                'best_val_score': best_val_score,
                'best_val_metrics': best_val_metrics,
                'best_path': best_path,
                'training_logs': training_logs,
                'val_steps': VAL_STEPS,
                'scheduler': scheduler,
                'world_size': world_size,
            }

        losses = train_epoch(
            model, train_loader, optimizer, task_criterion, device,
            GRAD_ACCUM_STEPS, STEPS_PER_LOG, epoch, rank,
            use_amp=use_amp, scaler=scaler,
            val_state=val_state
        )

        if val_state is not None:
            best_val_score = val_state['best_val_score']
            best_val_metrics = val_state['best_val_metrics']

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            eval_model = model.module if world_size > 1 else model

            val_metrics = evaluate(eval_model, val_loader, task_criterion, device, THRESHOLD)
            val_score = model_selection_score(val_metrics, CHECKPOINT_MODE)

            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"  Train Loss: {losses['combined']:.4f} "
                  f"(task={losses['task']:.4f}, act={losses['act']:.4f}, "
                  f"msg_decorr={losses['msg_decorr']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            if scheduler is not None:
                current_lrs = [g['lr'] for g in optimizer.param_groups]
                print(f"  LRs: enc={current_lrs[0]:.2e}, act={current_lrs[1]:.2e}, pipe={current_lrs[2]:.2e}")
            print()

            training_logs.append({
                'epoch': epoch,
                'train_loss_combined': losses['combined'],
                'train_loss_task': losses['task'],
                'train_loss_act': losses['act'],
                'train_loss_msg_decorr': losses['msg_decorr'],
                'val_loss': val_metrics['loss'],
                'val_roc_auc': val_metrics['roc_auc'],
                'val_micro_f1': val_metrics['micro_f1'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_selection_score': val_score
            })

            if val_score > best_val_score:
                best_val_score = float(val_score)
                best_val_metrics = {
                    'loss': float(val_metrics['loss']),
                    'roc_auc': float(val_metrics['roc_auc']),
                    'micro_f1': float(val_metrics['micro_f1']),
                    'macro_f1': float(val_metrics['macro_f1'])
                }
                torch.save({'model': eval_model.state_dict()}, best_path)
                print(f"[INFO] Saved best checkpoint ({CHECKPOINT_MODE}={val_score:.4f})")

        if scheduler is not None:
            if world_size > 1:
                score_tensor = torch.tensor([val_score if rank == 0 else 0.0], device=device)
                dist.broadcast(score_tensor, src=0)
                scheduler.step(score_tensor.item())
            else:
                scheduler.step(val_score)

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print("Final evaluation on test set")
        print(f"{'='*60}\n")

        eval_model = model.module if world_size > 1 else model

        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            eval_model.load_state_dict(ckpt["model"])
            eval_model.to(device)

        test_metrics = evaluate(eval_model, test_loader, task_criterion, device, THRESHOLD)
        val_metrics = evaluate(eval_model, val_loader, task_criterion, device, THRESHOLD)

        print("Validation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print()
        print("Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  Micro F1: {test_metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print()

        with open(out_dir / "training_logs.json", 'w') as f:
            json.dump(training_logs, f, indent=2)

        final_results = {
            'checkpoint_mode': CHECKPOINT_MODE,
            'best_val_selection_score': best_val_score,
            'best_val_metrics_at_selection': best_val_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        with open(out_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)

    if world_size > 1:
        dist.destroy_process_group()


def main():
    set_seed(SEED)

    print("Checking cache directory...")
    check_cache_dir(CACHE_DIR)
    print(f"Cache directory {CACHE_DIR} is writable.\n")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    print("Tokenizer loaded.\n")

    print(f"Pre-loading model to cache from {MODEL_NAME}...")
    _ = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    print("Model cached successfully.\n")

    print("Loading data...")
    train_data = load_json_data(TRAIN_JSON)
    val_data = load_json_data(VAL_JSON)
    test_data = load_json_data(TEST_JSON)
    print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.\n")

    print("Tokenizing and saving data...")
    tokenize_and_save_split(train_data, tokenizer, "train", out_dir, MAX_LENGTH)
    tokenize_and_save_split(val_data, tokenizer, "val", out_dir, MAX_LENGTH)
    tokenize_and_save_split(test_data, tokenizer, "test", out_dir, MAX_LENGTH)
    print("Tokenization complete.\n")

    pos_weight = compute_class_weight(train_data)
    print(f"Computed pos_weight: {pos_weight:.4f}\n")

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)\n")

    args = {
        'train_json': TRAIN_JSON,
        'val_json': VAL_JSON,
        'test_json': TEST_JSON,
        'pos_weight': pos_weight,
        'out_dir': out_dir
    }

    if num_gpus > 1:
        print(f"Using multi-GPU training with {num_gpus} GPUs\n")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        mp.spawn(train_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        print("Using single-GPU training\n")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
