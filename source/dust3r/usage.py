from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import numpy as np

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    #model_pt = "./checkpoints/dust3r_demo_224/checkpoint-best.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    classification = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'] #
    path = '../../../7SCENES'
    for class_name in classification:
        joined_path = os.path.join(path, class_name, 'test')
        if not os.path.exists(joined_path):
            raise FileNotFoundError(f"Directory {joined_path} does not exist. Please check the path.")
        path_dir = sorted([os.path.join(joined_path, x) for x in os.listdir(joined_path)])
        for p in path_dir:
            path_final = sorted([os.path.join(p, x) for x in os.listdir(p) if x.endswith('.color.png') and x not in 'frame-000000.color.png'])
            path_compare = p + '/frame-000000.color.png'
            T0 = np.loadtxt(os.path.join(p, 'frame-000000.pose.txt'))
            T0 = T0.reshape(4, 4)
            for pf in path_final:
                images = load_images([path_compare, pf], size=512)
                pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                output = inference(pairs, model, device, batch_size=batch_size)
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
                imgs = scene.imgs
                poses = scene.get_im_poses()
                T0_shift = poses[0].squeeze(0).detach().cpu().numpy()
                W = T0 @ np.linalg.inv(T0_shift)
                pose = poses[1::].squeeze(0).detach().cpu().numpy()
                pose = W @ pose
                key = pf[:-len('.color.png')]
                x = key + '.pose.txt'
                np.savetxt(x, pose, fmt='%.7e')
