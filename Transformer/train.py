import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

from dataset import SevenScenesRefine
from model   import RGBDRefineTransformerLarge
from utility import se3_exp

# ====== Config ====== #
DATA_DIR     = "../7SCENES"
PREDICT_DIR  = "../predict_pose"
CKPT_DIR     = "./checkpoints"
RESUME_CKPT  = None  # Path to a checkpoint to resume training, or None to start from scratch
Path(CKPT_DIR).mkdir(exist_ok=True, parents=True)

train_ratio   = 0.9
learning_rate = 1e-3
w_decay       = 1e-4
num_epochs    = 100
batch_size    = 24
warmup_epochs = 5

# Loss function weights
TRANSLATION_WEIGHT = 10.0  # Weight for translation loss
ROTATION_WEIGHT    = 5.0   # Weight for rotation loss

# ====== Dataset & Split ====== #
def train():
    full_set = SevenScenesRefine(DATA_DIR, PREDICT_DIR, train=True)
    idx      = np.random.RandomState(42).permutation(len(full_set))
    split    = int(train_ratio * len(full_set))
    train_set = Subset(full_set, idx[:split])
    valid_set = Subset(full_set, idx[split:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=10, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train: {len(train_set)} | Val: {len(valid_set)}")

    # ====== Model & Opt ====== #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = RGBDRefineTransformerLarge(pretrained=True)
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        ckpt = torch.load(resume, map_location='cpu')
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
        print(f"Resuming from {resume} at epoch {ckpt['epoch']}")
    else:
        print("Training from scratch")

    # Set up decay and no-decay parameter groups
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {'params': decay, 'weight_decay': w_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

    # Initialize optimizer
    opt    = optim.AdamW(param_groups, lr=learning_rate)
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        opt.load_state_dict(ckpt['opt'])
        print(f"Resuming optimizer state from {resume}")

    
    scaler = GradScaler()
    
    scheduler_cos = CosineAnnealingLR(opt, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler_warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[scheduler_warm, scheduler_cos],
        milestones=[warmup_epochs]
    )

    model.to(device)

    def geodesic(R):
        return torch.acos(torch.clamp(
            (torch.diagonal(R,0,-2,-1).sum(-1) - 1)/2, -1+1e-6, 1-1e-6))

    # ====== Train loop ====== #
    best_trans = float('inf')
    best_rot  = float('inf')

    for epoch in range(num_epochs):
        # --- training ---
        model.train()
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        for batch in tqdm(train_loader):
            rgb0, d0, rgbi, di = [batch[k].to(device, non_blocking=True)
                                for k in ('rgb0','d0','rgbi','di')]
            pose0, posei, pose_gt = [batch[k].to(device) for k in ('p0','pi','pgti')]
            fid  = batch['fid'].to(device)
            fid_emb_dim = model.embed_dim
            fid_emb = torch.sin(torch.arange(0,fid_emb_dim, device=device)[None,:] * fid[:,None] / 1000)

            with autocast():
                delta   = model(rgb0, d0, rgbi, di, fid_emb)   # (B,6)
                T_ref   = se3_exp(delta) @ posei               # (B,4,4)

                R_err = geodesic(T_ref[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                t_err = (T_ref[:,:3,3] - pose_gt[:,:3,3]).norm(dim=-1)
                loss  = ROTATION_WEIGHT * R_err.mean() + TRANSLATION_WEIGHT * t_err.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)                              # restore scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(opt)
            opt.zero_grad(set_to_none=True)
            scaler.update()

        # --- validation ---
        model.eval()
        tot_rb = 0
        tot_ra = 0
        tot_tb = 0
        tot_ta = 0
        cnt    = 0

        

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                rgb0,d0,rgbi,di = [batch[k].to(device) for k in ('rgb0','d0','rgbi','di')]
                posei, pose_gt = [batch[k].to(device) for k in ('pi','pgti')]
                fid = batch['fid'].to(device)
                fid_emb_dim = model.embed_dim
                fid_emb = torch.sin(torch.arange(0,fid_emb_dim, device=device)[None,:]*fid[:,None]/1000)

                delta = model(rgb0,d0,rgbi,di,fid_emb)
                T_ref = se3_exp(delta) @ posei

                rb = geodesic(posei[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                ra = geodesic(T_ref[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                tb = (posei[:,:3,3]-pose_gt[:,:3,3]).norm(dim=-1)
                ta = (T_ref[:,:3,3]-pose_gt[:,:3,3]).norm(dim=-1)

                tot_rb+=rb.sum()
                tot_ra+=ra.sum()
                tot_tb+=tb.sum()
                tot_ta+=ta.sum()

                cnt+=len(rb)
        print(f"Loss: {loss.item():.4f}, Learning rate: {opt.param_groups[0]['lr']:.6f}")
        print(f"VALIDATION: trans {tot_tb/cnt*100:.2f}->{tot_ta/cnt*100:.2f} cm | "
              f"rot {tot_rb/cnt*57.3:.2f}°->{tot_ra/cnt*57.3:.2f}°")
        scheduler.step()

        # --- checkpoint ---
        torch.save({'epoch':epoch,
                    'model':model.state_dict(),
                    'opt'  :opt.state_dict()},
                f"{CKPT_DIR}/epoch{epoch:03d}.pth")
        if tot_ta < best_trans and tot_ra < best_rot:
            print(f"New best model at epoch {epoch+1}, saving...")
            best_trans = tot_ta
            best_rot = tot_ra
            torch.save({'epoch':epoch,
                        'model':model.state_dict(),
                        'opt'  :opt.state_dict()},
                    f"{CKPT_DIR}/best.pth")


if __name__ == '__main__':
    train()