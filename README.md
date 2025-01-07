
# A Semi-Supervised Fracture-Attention Model for Segmenting Tubular Objects with Improved Topological Connectivity

This is the official code of [A Semi-Supervised Fracture-Attention Model for Segmenting Tubular Objects with Improved Topological Connectivity](https://). (Bioinformatics 2025.01)

## Our Contributions
**- Semi-Supervised Fracture-Attention Model (SSFA)**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SSFA/blob/main/figure/Overview.png" width="100%" >
<br>Overview of SSFA.
</p>

**- More Intuitive Topological Evaluation Metric: Fracture Rate (FR)**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SSFA/blob/main/figure/Fracture%20Rate.png" width="50%" >
<br>Visualization of fractures in segmentation results.
</p>

$$
FR=\frac{N_F}{N_Y}\times 100\%
$$

## Quantitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SSFA/blob/main/figure/Quantitative%20Comparison.png" width="100%" >
</p>


## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SSFA/blob/main/figure/Qualitative%20Comparison.png" width="100%" >
<br>Qualitative results on ER, SNEMI3D, CREMI and STARE-DRIVE. (a) Raw images. (b) Ground truth. (c) UNet. (d) TopoLoss. (e) clDice. (f) MT. (g) CPS. (h) SSFA. Red arrows highlight differences among the results.
</p>


## Requirements
```
albumentations==0.5.2  
matplotlib==3.2.1  
networkx==2.3  
numba==0.44.1  
numpy==1.18.5  
opencv_python==4.2.0.32  
Pillow==9.3.0  
ripser==0.6.4  
scikit_image==0.19.1  
scikit_learn==1.2.0  
scipy==1.4.1  
skimage==0.0  
torch==1.8.0  
torchvision==0.9.0  
tqdm==4.32.1  
visdom==0.1.8.9
```

## Usage
**Data preparation**
Your datasets directory tree should be look like this:
>to see [tools/attention_map/initial_attention.py](https://) for **structure1**, **structure2**, **semantic**, **attention**.
```
dataset
├── train_sup
    ├── image
        ├── 1.tif
        └── ...
    ├── structure1
        ├── 1.tif
        └── ...
    ├── structure2
        ├── 1.tif
        └── ...
    ├── semantic
        ├── 1.tif
        └── ...
    ├── attention
        ├── 1.tif
        └── ...
    └── mask
        ├── 1.tif
        └── ...
├── train_unsup
    ├── image
    ├── structure1
    ├── structure2
    ├── semantic
    └── attention
├── val
    ├── image
    ├── structure1
    ├── structure2
    ├── semantic
    ├── attention
    └── mask
```
**Training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_semi_TA.py
```
**Testing**
```
python -m torch.distributed.launch --nproc_per_node=4 test_TA.py
```

## Citation
If our work is useful for your research, please cite our paper:
```
```

