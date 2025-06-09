# 113_2 Computer Vision Final Project - Microsoft 3D Reconstruction

## Links
1. GitHub Repository 

    [Link](https://github.com/lsy1205/113_2_CV_Final.git)
    ```bash 
    git clone https://github.com/lsy1205/113_2_CV_Final.git
    ```
2. Report

    [Link](https://docs.google.com/document/d/1bfUDjlT-mH5iGmYgFJJ8f5ZCTbWA8-vZiz1gJkuIbnU/edit?usp=sharing)

## Group: 3d_macarons 

## Members:
* R13942173 彭柏翔
* R13921040 劉瑄穎
* R12521530 游文歆

## Environment Setup:
We recommend skipping the environment setup for **Fast3R**. 
### DUSt3R
1. Create the environment, here we show an example using conda.
``` bash
cd ./source/dust3r
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
cd ../../../../
```
3. Install Timm and other packages for Customized Transformer, Download, and 3D Reconstruction
``` bash
pip install timm 
pip install gdown
pip install open3d
pip install plyfile
```

### Fast3R (Optional)

Our provided test files do **not** use Fast3R to generate camera poses.  
This section is only relevant if you wish to experiment with Fast3R-based pose estimation.

1. Installation
```bash
cd ./source/fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt

# install fast3r as a package (so you can import fast3r and use it in your own project)
pip install -e .
cd ../
```
2. Install Timm and other packages for Customized Transformer, Download, and 3D Reconstruction
``` bash
pip install timm 
pip install gdown
pip install open3d
pip install plyfile
```

## Results
* Accuracy = 0.0, Completeness = 0.51 <br>
The test result is in the folder *test*

* Accuracy = 0.18, Completeness = 0.05 <br>
The test result is in the folder *test_dust* <br>
The bonus result is in the folder *bonus* (We choose this method to generate our bonus result)

* Accuracy = 0.16, Completeness = 0.1 <br>
The test result is in the folder *test_dust_transformer*

To get the **test** point clouds and the **bonus** point clouds, you can run the command below.

```bash
bash get_bonus.sh
bash get_tests.sh
```

## 3D Reconstruction
In this section, we demonstrate how to reproduce our results step by step.

We will walk through the process of generating the three reconstruction results described earlier in the [Results](#results) section.

**Please activate conda env dust3r for the following steps**
```bash
conda activate dust3r 
```

### Preprocess
Please place the **7SCENES** dataset folder in the root directory of this repository — that is, at the same level as the `source` directory.
```bash
repo_root/
├── 7SCENES/
├── bonus
├── source/
├── test
└── README.md
```

The **7SCENES** dataset should follow the directory structure shown below.

For **test** sequences, only the pose of **frame 0** is required.  
For **train** sequences, poses for **all frames** should be provided.

```bash
├── chess
│   ├── test
│   │   ├── seq-xx
│   │   │   ├── frame-xxxxxx.color.png
│   │   │   ├── frame-xxxxxx.depth.png
│   │   │   └── frame-xxxxxx.depth.proj.png
│   │   └── sparse-seq-xx
│   └── train
│       └── seq-xx
│           ├── frame-xxxxxx.color.png
│           ├── frame-xxxxxx.depth.png
│           ├── frame-xxxxxx.depth.proj.png
│           └── frame-xxxxxx.pose.txt
├── fire
├── heads
├── office
├── pumpkin
├── redkitchen
└── stairs
```

### Reproducing the 3 Results
We recommend skipping all the **optional** steps in this section, as they are typically time-consuming.

1. Directly use seq2ply to reconstruct the 3d point cloud.<br>
**Accuracy = 0.0, Completeness = 0.51**
    ```bash
    cd ./source
    bash generate_result1.sh
    cd ../
    ```

2. Inference on DUSt3R and predict the pose.txt of each picture<br>
Then utilize the matrices transformation to get a more precise position.<br>
**Accuracy = 0.18, Completeness = 0.05<br>**
    (1) Go into the `source` directory
    ```bash
    cd ./source
    ```
    (2) (Optional) Generate poses using DUSt3R. 
    
    This step is **optional**. We have already provided the generated poses in the `pose_dust` directory.

    Note that this process is quite time-consuming.


    ```bash
    cd ./dust3r/
    python usage.py
    cd ../
    ```

    If you choose to run this step, please make sure to update the `POSE_DIR` variable in `generate_result2.sh` accordingly.  
    Specifically, the `predict_path` in `usage.py` must match the `POSE_DIR` specified in `generate_result2.sh`.

    You also need to copy the ground truth pose of frame 0 into the `predict_path` directory.  
    Make sure that the `DEST_DIR` variable in `copy_frame0.sh` matches the `predict_path` specified in `usage.py`.

    ```bash
    bash copy_frame0.sh
    ```


    (3) Generate the point clouds with the predicted poses.
    ```bash
    bash generate_result2.sh
    cd ../
    ```

3. Use DUSt3R to generate initial pose predictions, applied post-processing, and then refined the results using a transformer-based model.<br>
**Accuracy = 0.16, Completeness = 0.1**  
    (1) Go into the `source` directory
    ```bash
    cd ./source
    ```

    (2) (Optional) Refine DUSt3R Poses Using a Customized Transformer. 

    This step is optional, as we have already provided the refined poses in the `refined_pose` directory.

    If you wish to perform the refinement yourself, please run the shell script below.

    ```bash
    cd ./Transformer
    bash get_checkpoint.sh
    bash inference.sh
    cd ../
    ```

    The input to the transformer is the `pose_dust` directory we provide by default.  
    If you generated the poses yourself (i.e., by running step (2) in the previous section),  
    you can update the `POSE_DIR` variable in the `inference.sh` script accordingly.


    (3) Generate the point clouds with the refined poses.
    
    ```bash
    bash generate_result3.sh
    cd ../
    ```
### Reproducing the Bonus Point Clouds
```bash
cd ./source
bash generate_bonus.sh
cd ../
```

### Training the Customized Transformer
All of the customized transformer files are placed in the `./source/Transformer` folder.

To train the customized transformer, you can run the shell script below.
```bash
cd ./source/Transformer
python train.py
```

Here, we use the predicted poses provided in the `pose_fast` folder.  
If you would like to train using your own predicted poses, please follow the same directory structure as used in `pose_fast`.

## Other Utility Files
* `color_image_to_video.py`

    This script converts color images into an MP4 video file.  
    We use the generated video to observe the characteristics of the image sequence.  
    Through this, we found that dense sequences exhibit continuous and consistent color frames.

    **Example Usage:**
    ```bash
    python color_image_to_video.py --input ../7SCENES/chess/test/seq-03 --output test.mp4
    ```

* `ICP_correction.py`

    This script performs ICP correction on the predicted poses.  
    You can refer to `generate_icp.sh` for an example of how to use it.

    **Example Usage:**

    ```bash
    python ICP_correction.py --color_dir [path to color image folder] \
                             --depth_dir [path to depth image folder] \
                             --pred_pose_dir [path to predicted pose folder] \
                             --out_pose_dir [path to the output folder]
    ```

* `seq2ply.py`

    This script reconstructs 3D point clouds using color images, depth images, and ground truth poses.

    **Example Usage:**
    ```bash
    python seq2ply.py --seq_path [path to the sequence] \
                    --ply_path [path to the output 3D point cloud]
    ```

* `seq2ply_pred.py`  
    This script reconstructs 3D point clouds using color images, depth images, and predicted poses.  
    The main difference from `seq2ply.py` is that this script includes a parameter to specify the path to the predicted poses.
    **Example Usage:**
    ```bash
    python seq2ply_pred.py --seq_path [path to the sequence] \
                           --predict_pose_path [path to the predicted pose] \
                           --ply_path [path to the output 3D point cloud] \
    ```

* `utils.py`
    This script is used to evaluate reconstructed point clouds against the ground truth.  
    You can refer to `score.sh` for an example of how to use it.

    **Example Usage:**
    ```bash
    python utils.py --gt_ply [path to the ground truth point cloud] --rec_ply [path to the reconstructed point cloud]
    ```
