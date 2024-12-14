

<div align="center">
<a href="https://www.imagehub.cc/image/TUR-logo.CJAeR0"><img src="https://s1.imagehub.cc/images/2024/12/13/ba6b730d67ca5f72b38570315db39241.png" alt="TUR logo" border="0"></a>

# [AAAI 2025] Debiased All-in-one Image Restoration with Task Uncertainty Regularization

[Gang Wu (吴刚)](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang (江俊君)](http://homepage.hit.edu.cn/jiangjunjun), [Yijun Wang (王奕钧)](), [Kui Jiang (江奎)](https://github.com/kuijiang94), and [Xianming Liu (刘贤明)](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Faculty of Computing, Harbin Institute of Technology, Harbin 150001, China.

[![Paper](http://img.shields.io/badge/Paper-OpenReview-FF6B6B.svg)](https://openreview.net/forum?id=kx7eyKgEGz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DAAAI.org%2F2025%2FConference%2FAuthors%23your-submissions))
[![arXiv](https://img.shields.io/badge/AAAI-2025-red.svg)]()
[![Models](https://img.shields.io/badge/BaiduPan-Models-blue.svg)](https://pan.baidu.com/s/1YN3P-CmnisXVIdLHTWB9Fw?pwd=AAAI)
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAitical%2FTUR%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

</div>


## Overview

<div align="center"> <table> <tr> <td align="center"><img src="assets/framework.png" width="385"><br>Framework</td> <td align="center"><img src="assets/uncertainty.png" width="385"><br>Task Uncertainty</td> </tr> </table> </div>


>All-in-one image restoration is a fundamental low-level vision task with significant real-world applications. The primary challenge lies in  addressing diverse degradations within a single model. While current methods primarily exploit task prior information to guide the restoration models, they typically employ uniform multi-task learning, overlooking the heterogeneity in model optimization across different degradation tasks. To eliminate the bias, we propose a task-aware optimization strategy, that introduces adaptive task-specific regularization for multi-task image restoration learning. Specifically, our method dynamically weights and balances losses for different restoration tasks during training, encouraging the implementation of the most reasonable optimization route. In this way, we can achieve more robust and effective model training. Notably, our approach can serve as a plug-and-play strategy to enhance existing models without requiring modifications during inference. Extensive experiments across diverse all-in-one restoration settings demonstrate the superiority and generalization of our approach. For instance, AirNet retrained with TUR achieves average improvements of **1.3 dB** on three distinct tasks and **1.81 dB** on five distinct all-in-one tasks. These results underscore TUR's effectiveness in advancing the SOTAs in all-in-one image restoration, paving the way for more robust and versatile image restoration.


<div align="center"> <table> <tr> <td align="center"><a href="https://imgsli.com/MzI3MzEz"><img src="assets/transweather.png" width="385"></a></td> <td align="center"><a href="https://imgsli.com/MzI3MzE0"><img src="assets/outdoor-rain.png" width="385"></a></td> </tr> <tr> <td align="center"><a href="https://imgsli.com/MzI3MzEw"><img src="assets/airnet.png" width="385"></a></td> <td align="center"><a href="https://imgsli.com/MzI3MzE2"><img src="assets/promptir.png" width="385"></a></td> </tr> </table> <a href="https://www.imagehub.cc/image/intro-improve2.bkhl1O"> <img src="https://s1.imagehub.cc/images/2024/08/19/ba47d9f56f5b9b684c8faa2788c3dfeb.png" alt="intro improve2" border="0"> </a> </div>

## Results

Table: Comparison on 7 distinct degradation tasks introduced in [1].
<div align="center"> <table> <thead> <tr> <th>Methods</th> <th>SR</th> <th>Blur</th> <th>Noise</th> <th>JPEG</th> <th>Rain</th> <th>Haze</th> <th>Dark</th> <th>Avg.</th> </tr> </thead> <tbody> <tr> <td>SRResNet</td> <td>25.52</td> <td>30.01</td> <td>30.49</td> <td>32.46</td> <td>32.38</td> <td>25.57</td> <td>30.20</td> <td>29.52</td> </tr> <tr> <td><b>SRResNet (Ours)</b></td> <td><b>25.55</b></td> <td><b>30.65</b></td> <td><b>30.65</b></td> <td><b>32.92</b></td> <td><b>35.20</b></td> <td><b>26.16</b></td> <td><b>32.04</b></td> <td><b>30.45</b></td> </tr> </tbody> </table> </div>


Table: Comparisons under the three-degradation all-in-one setting: a unified model is trained on a combined set of images obtained from all degradation types and levels.
| Method | Dehazing on SOTS | Deraining on Rain100L | Denoising on BSD68 $\sigma=15$ | Denoising on BSD68 $\sigma=25$ | Denoising on BSD68 $\sigma=50$ | Average |
|--------|-------------------|------------------------|--------------------------------|--------------------------------|--------------------------------|---------|
| BRDNet [1] | 23.23/0.895 | 27.42/0.895 | 32.26/0.898 | 29.76/0.836 | 26.34/0.693 | 27.80/0.843 |
| LPNet [2] | 20.84/0.828 | 24.88/0.784 | 26.47/0.778 | 24.77/0.748 | 21.26/0.552 | 23.64/0.738 |
| FDGAN [3] | 24.71/0.929 | 29.89/0.933 | 30.25/0.910 | 28.81/0.868 | 26.43/0.776 | 28.02/0.883 |
| MPRNet [4] | 25.28/0.955 | 33.57/0.954 | 33.54/0.927 | 30.89/0.880 | 27.56/0.779 | 30.17/0.899 |
| DL [5] | 26.92/0.931 | 32.62/0.931 | 33.05/0.914 | 30.41/0.861 | 26.90/0.740 | 29.98/0.876 |
| AirNet [6] | 27.94/0.962 | 34.90/0.968 | 33.92/0.933 | 31.26/0.888 | 28.00/0.797 | 31.20/0.910 |
| **AirNet (Ours)** | **33.49/0.976** | **38.70/0.984** | **33.96/0.931** | **31.31/0.886** | **28.04/0.794** | **32.50/0.914** |
| PromptIR [7] | 30.58/0.974 | 36.37/0.972 | 33.98/0.933 | 31.31/0.888 | 28.06/0.799 | 32.06/0.913 |
| **PromptIR (Ours)** | **31.17/0.978** | **38.57/0.984** | **34.06/0.932** | **31.40/0.887** | **28.13/0.797** | **32.67/0.916** |


Table: Comparative results on five distinct tasks in all-in-one image restoration.

| Method | Dehazing on SOTS | Deraining on Rain100L | Denoising on BSD68 | Deblurring on GoPro | Low-Light on LOL | Average |
|--------|-------------------|------------------------|---------------------|----------------------|-------------------|---------|
| NAFNet [1] | 25.23/0.939 | 35.56/0.967 | 31.02/0.883 | 26.53/0.808 | 20.49/0.809 | 27.76/0.881 |
| MPRNet [2] | 24.27/0.937 | 38.16/0.981 | 31.35/0.889 | 26.87/0.823 | 20.84/0.824 | 28.27/0.890 |
| SwinIR [3] | 21.50/0.891 | 30.78/0.923 | 30.59/0.868 | 24.52/0.773 | 17.81/0.723 | 25.04/0.835 |
| DL [4] | 20.54/0.826 | 21.96/0.762 | 23.09/0.745 | 19.86/0.672 | 19.83/0.712 | 21.05/0.743 |
| TAPE [5] | 22.16/0.861 | 29.67/0.904 | 30.18/0.855 | 24.47/0.763 | 18.97/0.621 | 25.09/0.801 |
| IDR [6] | 25.24/0.943 | 35.63/0.965 | 31.60/0.887 | 27.87/0.846 | 21.34/0.826 | 28.34/0.893 |
| Transweather [7] | 21.32/0.885 | 29.43/0.905 | 29.00/0.841 | 25.12/0.757 | 21.21/0.792 | 25.22/0.836 |
| **Transweather (Ours)** | **29.68/0.966** | **33.09/0.952** | **30.40/0.869** | **26.63/0.815** | **23.02/0.838** | **28.56/0.888** |
| AirNet [8] | 21.04/0.884 | 32.98/0.951 | 30.91/0.882 | 24.35/0.781 | 18.18/0.735 | 25.49/0.846 |
| **AirNet (Ours)** | **27.59/0.954** | **33.95/0.962** | **30.93/0.875** | **26.13/0.801** | **17.88/0.772** | **27.30/0.873** |

Table: Comparison on deweathering tasks with Allweather dataset [3].

| Datasets | Method | PSNR ↑ | SSIM ↑ |
|----------|--------|--------|--------|
| Outdoor-Rain | All-in-One [1] | 24.71 | 0.8980 |
| | WeatherDiff₁₂₈ [2] | 29.72 | 0.9216 |
| | TransWeather [3] | 28.83 | 0.9000 |
| | **TransWeather (Ours)** | **29.75** | **0.9073** |
| Snow100K | DDMSNet [4] | 28.85 | 0.8772 |
| | All-in-One [1] | 28.33 | 0.8820 |
| | WeatherDiff₁₂₈ [2] | 29.58 | 0.8941 |
| | TransWeather [3] | 29.31 | 0.8879 |
| | **TransWeather (Ours)** | **30.62** | **0.9086** |
| RainDrop | All-in-One [1] | 31.12 | 0.9268 |
| | WeatherDiff₁₂₈ [2] | 29.66 | 0.9225 |
| | TransWeather [3] | 30.17 | 0.9157 |
| | **TransWeather (Ours)** | **31.61** | **0.9330** |
| Average | All-in-One [1] | 27.12 | 0.8933 |
| | WeatherDiff₁₂₈ [2] | 29.65 | 0.9127 |
| | TransWeather [3] | 29.44 | 0.9012 |
| | **TransWeather (Ours)** | **30.66** | **0.9163** |


Table: Comparison on de-weathering tasks on real-world datasets following [2]
| Datasets | Method | PSNR ↑ | SSIM ↑ |
|----------|--------|--------|--------|
| SPA+ | Chen et al. [1] | 37.32 | 0.97 |
| | WGWSNet [2] | 38.94 | 0.98 |
| | TransWeather [3] | 33.64 | 0.93 |
| | **TransWeather (Ours)** | **39.78** | **0.98** |
| RealSnow | Chen et al. [1] | 29.37 | 0.88 |
| | WGWSNet [2] | 29.46 | 0.85 |
| | TransWeather [3] | 29.16 | 0.82 |
| | **TransWeather (Ours)** | **29.72** | **0.91** |
| REVIDE | Chen et al. [1] | 20.10 | 0.85 |
| | WGWSNet [2] | 20.44 | 0.87 |
| | TransWeather [3] | 17.33 | 0.82 |
| | **TransWeather (Ours)** | **20.38** | **0.88** |
| Average | Chen et al. [1] | 28.93 | 0.90 |
| | WGWSNet [2] | 29.61 | 0.90 |
| | TransWeather [3] | 26.71 | 0.86 |
| | **TransWeather (Ours)** | **29.96** | **0.92** |

## Citation
If our project helps your research or work, please cite our paper or star this repo. Thank you!
```
@inproceedings{wu2025debiased,
  title={Debiased All-in-one Image Restoration with Task Uncertainty Regularization},
  author={Gang Wu, Junjun Jiang, Yijun Wang, Kui Jiang, and Xianming Liu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowledgement

This project is based on [MioIR](https://github.com/Xiangtaokong/MiOIR/tree/main/basicsr), thanks for their nice sharing.
