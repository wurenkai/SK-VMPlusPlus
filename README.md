<p align="center">
  <h1 align="center">SK-VMPlusPlus</h1>
  <h3 align="center">Mamba Assists Skip-connections for Medical Image Segmentation</h3>
  <p align="center">
    Renkai Wu, Liuyue Pan, Pengchen Liang, Qing Chang, Xianjin Wang*, Weihuan Fang*
  </p>
    <p align="center">
      1. Department of General Surgery, Shanghai Key Laboratory of Gastric Neoplasms, Shanghai Institute of Digestive Surgery, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine</br>
      2. Department of Radiology, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine</br>
      3. Department of Urology, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine</br>
      4. School of Microelectronics, Shanghai University</br>
      5. Department of General Medicine, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine</br>
  </p>
</p>

**0. Main Environments.** </br>
```
conda create -n SKVMPlusPlus python=3.8
conda activate SKVMPlusPlus
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.**

*A.ISIC2017* </br>
1- Download the ISIC 2017 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2- Run `Prepare_ISIC2017.py` for data preparation and dividing data to train,validation and test sets. </br>

*B.CVC-ClinicDB* </br>
1- Download the CVC-ClinicDB dataset from [this](https://polyp.grand-challenge.org/CVCClinicDB/) link. </br>

*C.Promise12* </br>
1- Download the Promise12 dataset from [this](https://promise12.grand-challenge.org/) link. </br>

*D.UWF-RHS* </br>
1- Download the UWF-RHS dataset from [this](https://github.com/wurenkai/UWF-RHS-Dataset-and-MASNet) link. </br>

*D. Prepare your own dataset* </br>
1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './your_dataset/'
  - images
    - 0000.png
    - 0001.png
  - masks
    - 0000.png
    - 0001.png
  - Prepare_your_dataset.py
2. In the 'Prepare_your_dataset.py' file, change the number of training sets, validation sets and test sets you want.</br>
3. Run 'Prepare_your_dataset.py'. </br>

**2. Train the SKVMPlusPlus.** </br>
First, download Res2Net-50 pre-training weights at this [link](https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth). Then,
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>

**3. Test the SKVMPlusPlus.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>


## Citation
If you find this repository helpful, please consider citing: </br>
```
@article{WU2025107646,
  title = {SK-VM++: Mamba assists skip-connections for medical image segmentation},
  author = {Renkai Wu and Liuyue Pan and Pengchen Liang and Qing Chang and Xianjin Wang and Weihuan Fang},
  journal = {Biomedical Signal Processing and Control},
  volume = {105},
  pages = {107646},
  year={2025},
  publisher={Elsevier}
}
```
## Acknowledgement
Thanks to [Ultralight VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) and [MSNet](https://github.com/Xiaoqi-Zhao-DLUT/MSNet-M2SNet) for their outstanding work.

