# PyTorch Demo of the Hyperspectral Image Classification method - MAR_LWFormer.

Using the code should cite the following paper:

Y. Fang, Q. Ye, L. Sun, Y. Zheng and Z. Wu, "Multiattention Joint Convolution Feature Representation With Lightweight Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-14, 2023, Art no. 5513814, doi: 10.1109/TGRS.2023.3281511.

@ARTICLE{10138605,
author={Fang, Yu and Ye, Qiaolin and Sun, Le and Zheng, Yuhui and Wu, Zebin},
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
title={Multiattention Joint Convolution Feature Representation With Lightweight Transformer for Hyperspectral Image Classification}, 
year={2023},
volume={61},
number={},
pages={1-14},
doi={10.1109/TGRS.2023.3281511}}

# Description.
Hyperspectral image (HSI) classification is currently a hot topic in the field of remote sensing. The goal is to utilize the spectral and spatial information from HSI to accurately identify land covers. Convolution neural network (CNN) is a powerful approach for HSI classification. However, CNN has limited ability to capture nonlocal information to represent complex features. Recently, vision transformers (ViTs) have gained attention due to their ability to process non-local information. Yet, under the HSI classification scenario with ultrasmall sample rates, the spectral–spatial information given to ViTs for global modeling is insufficient, resulting in limited classification capability. Therefore, in this article, multiattention joint convolution feature representation with lightweight transformer (MAR-LWFormer) is proposed, which effectively combines the spectral and spatial features of HSI to achieve efficient classification performance at ultrasmall sample rates. Specifically, we use a three-branch network architecture to extract multiscale convolved 3D-CNN, extended morphological attribute profile (EMAP), and local binary pattern (LBP) features of HSI, respectively, by taking full exploitation of ultrasmall training samples. Second, we design a series of multiattention modules to enhance spectral–spatial representation for the three types of features and to improve the coupling and fusion of multiple features. Third, we propose an explicit feature attention tokenizer (EFA-tokenizer) to transform the feature information, which maximizes the effective spectral–spatial information retained in the flat tokens. Finally, the generated tokens are input to the designed lightweight transformer for encoding and classification. Experimental results on three datasets validate that MAR-LWFormer has an excellent performance in HSI classification at ultrasmall sample rates when compared to several state-of-the-art classifiers.
