import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

from dataset import SevenScenesRefine
from model   import RGBDRefineTransformer
from utility import se3_exp

# ====== Config ====== #
DATA_DIR     = "../7SCENES"
PREDICT_DIR  = "../predict_pose_no_calibration"
CKPT_DIR     = "./checkpoints"
Path(CKPT_DIR).mkdir(exist_ok=True, parents=True)

train_ratio  = 0.9
lr           = 2e-4
w_decay      = 1e-4
num_epochs   = 100
batch_size   = 32 

# ====== Dataset & Split ====== #
def train(resume=False):
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
    model  = RGBDRefineTransformer()
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
        print(f"Resuming from {resume} at epoch {ckpt['epoch']}")
    else:
        print("Training from scratch")
    model.to(device)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)
    scaler = GradScaler()

    def geodesic(R):
        return torch.acos(torch.clamp(
            (torch.diagonal(R,0,-2,-1).sum(-1) - 1)/2, -1+1e-6, 1-1e-6))

    # ====== Train loop ====== #
    for epoch in range(num_epochs):
        # --- training ---
        model.train()
        pbar = tqdm(train_loader, desc=f'E{epoch:02d}')
        for batch in pbar:
            rgb0, d0, rgbi, di = [batch[k].to(device, non_blocking=True)
                                for k in ('rgb0','d0','rgbi','di')]
            pose0, posei, pose_gt = [batch[k].to(device) for k in ('p0','pi','pgti')]
            fid  = batch['fid'].to(device)
            fid_emb = torch.sin(torch.arange(0,384, device=device)[None,:] * fid[:,None] / 1000)

            with autocast():
                delta   = model(rgb0, d0, rgbi, di, fid_emb)   # (B,6)
                T_ref   = se3_exp(delta) @ posei               # (B,4,4)

                R_err = geodesic(T_ref[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                t_err = (T_ref[:,:3,3] - pose_gt[:,:3,3]).norm(dim=-1)
                loss  = 10 * R_err.mean() + t_err.mean()

            scaler.scale(loss).backward()
            scaler.step(opt); opt.zero_grad(); scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # --- validation ---
        model.eval()
        tot_rb=tot_ra=tot_tb=tot_ta=cnt=0
        with torch.no_grad():
            for batch in valid_loader:
                rgb0,d0,rgbi,di = [batch[k].to(device) for k in ('rgb0','d0','rgbi','di')]
                posei, pose_gt = [batch[k].to(device) for k in ('pi','pgti')]
                fid = batch['fid'].to(device)
                fid_emb = torch.sin(torch.arange(0,384, device=device)[None,:]*fid[:,None]/1000)

                delta = model(rgb0,d0,rgbi,di,fid_emb)
                T_ref = se3_exp(delta) @ posei

                rb = geodesic(posei[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                ra = geodesic(T_ref[:,:3,:3] @ pose_gt[:,:3,:3].transpose(-1,-2))
                tb = (posei[:,:3,3]-pose_gt[:,:3,3]).norm(dim=-1)
                ta = (T_ref[:,:3,3]-pose_gt[:,:3,3]).norm(dim=-1)

                tot_rb+=rb.sum(); tot_ra+=ra.sum()
                tot_tb+=tb.sum(); tot_ta+=ta.sum()
                cnt+=len(rb)

        print(f"[VAL] trans {tot_tb/cnt*100:.2f}->{tot_ta/cnt*100:.2f} cm | "
            f"rot {tot_rb/cnt*57.3:.2f}->{tot_ra/cnt*57.3:.2f} °")

        # --- checkpoint ---
        torch.save({'epoch':epoch,
                    'model':model.state_dict(),
                    'opt'  :opt.state_dict()},
                f"{CKPT_DIR}/epoch{epoch:02d}.pth")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', default=None, type=str, help='Checkpoint to resume from')
    args = ap.parse_args()

    train(resume=args.resume)