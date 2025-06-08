import torch
def hat(v):               # (B,3) → (B,3,3)
    B = v.shape[0]
    x,y,z = v[:,0],v[:,1],v[:,2]
    O = torch.zeros_like(x)
    return torch.stack([O,-z,y, z,O,-x, -y,x,O],dim=1).reshape(B,3,3)

def se3_exp(xi):          # xi (B,6) → (B,4,4)
    w, v = xi[:,:3], xi[:,3:]
    θ = torch.linalg.norm(w,dim=1,keepdim=True)+1e-8
    A = torch.sin(θ)/θ
    B = (1-torch.cos(θ))/θ**2
    C = (1-A)/θ**2
    w_hat = hat(w)
    R = torch.eye(3,device=xi.device).unsqueeze(0)+A[:,:,None]*w_hat+B[:,:,None]*(w_hat@w_hat)
    V = torch.eye(3,device=xi.device).unsqueeze(0)+B[:,:,None]*w_hat+C[:,:,None]*(w_hat@w_hat)
    t = (V@v.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4,device=xi.device).unsqueeze(0).repeat(R.shape[0],1,1)
    T[:,:3,:3],T[:,:3,3]=R,t
    return T
