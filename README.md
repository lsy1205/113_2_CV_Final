# 113_2 Computer Vision Final Project - Microsoft 3D Reconstruction

## Group: 3d_macarons 

## Members:
R13942173 彭柏翔
R13921040 劉瑄穎
R12521530 游文歆

## Environment Setup:
### DUSt3R
1. Create the environment, here we show an example using conda.
``` bash
cd dust3r
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```
2. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
``` bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```
3. Install Timm for Customized Transformer
``` bash
pip install timm 
```

### Fast3R (Optional, Our test Files does not use Fast3R to generate poses)
1. Installation
```bash
cd fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt

# install fast3r as a package (so you can import fast3r and use it in your own project)
pip install -e .
```
2. Install Timm for Customized Transformer
``` bash
pip install timm 
```
## 3D Reconstruction : 
(Please refer to our GitHub README.md : https://github.com/lsy1205/113_2_CV_Final.git)
1. Directly use seq2ply to reconstruct the 3d point cloud<br>
Accuracy = 0.0, Completeness = 0.51
```bash

```

2. Inference on DUSt3R and predict the pose.txt of each picture<br>
Then utilize the matrices transformation to get a more precise position.
Accuracy = 0.18, Completeness = 0.05<br>
(p.s. assuming that the /7SCENES is as at the same level as /source file)
```bash
cd source/dust3r/
python usage.py
```

3. Employed DUSt3R with a post-processing step, followed by a transformer-based refinement of the predicted poses.<br>
Accuracy = 0.16, Completeness = 0.1
```bash

```

## Results
* Accuracy = 0.0, Completeness = 0.51 <br>
The test result is in the folder *test*

* Accuracy = 0.18, Completeness = 0.05 <br>
The test result is in the folder *test_dust* <br>
The bonus result is in the folder *bonus* (We choose this method to generate our bonus result)

* Accuracy = 0.16, Completeness = 0.1 <br>
The test result is in the folder *test_dust_transformer*
