# CLIP-MEI: Exploit more effective information for few-shot action recognition (KBS)
Official Pytorch Implementation of CLIP-MEI
> Exploit more effective information for few-shot action recognition (CLIP-MEI)
> 
> XuanHan Deng, WenZhu Yanga, XinBo zhao, Tong Zhou and Xin Deng

> Abstract: Few-shot action recognition (FSAR) aims to address the challenge of limited labeled data, yet most existing methods struggle to effectively tackle the insufficient visual information arising from scarce labeled samples. So, this paper proposes the CLIP-MEI framework, which enhances model representation through multi-modal feature fusion and latent information mining. Specifically, we build a CLIP-based prototype matching framework and design three core modules: 1) Query-Specific Semantic Information Augmentation (QSA), which generates adaptive semantic embeddings by integrating support set label semantics with query visual features to mitigate semantic disparities between support and query sets; 2) Task-based Feature Enhancement (TFE), which optimizes feature representations by exploiting latent relationships between support and query sets within the same task; and 3) Motion Information Compensation (MIC), which extracts highly invariant motion features by aligning shallow and deep motion representations. Extensive results demonstrate that CLIP-MEI establishes new performance records across diverse benchmark datasets, notably achieving leading results on HMDB51. For example, it attains a 1-shot accuracy of 76.4\% on HMDB51, outperforming baselines by 10.1\%. The implementation can be accessed via [GitHub](https://github.com/D-XH/CLIP-MEI.git)

This code is based on [TRX](https://github.com/tobyperrett/trx) codebase.

This paper has been accepted by Knowledge based systems. ([https://doi.org/10.1016/j.knosys.2025.113965](https://doi.org/10.1016/j.knosys.2025.113965))

# Results

## CLIP-RN50
| - | 1-shot | 3-shot | 5-shot |
| ---- | ---- | ---- | ---- |
| HMDB51 | 76.4 | 83.0 | 84.9 |
| SSv2-FULL | 61.6 | 65.9 | 68.5 |
| UCF101 | 93.7 | 97.0 | 97.5 |
| Kinetics | 90.1 | 92.9 | 93.6 |

## CLIP-VIT
| - | 1-shot | 3-shot | 5-shot |
| ---- | ---- | ---- | ---- |
| HMDB51 | 75.9 | 85.9 | 88.5 |
| SSv2-FULL | 67.8 | 76.6 | 79.8 |
| UCF101 | 94.9 | 98.2 | 98.9 |
| Kinetics | 87.7 | 95.3 | 96.0 |

# Data preparation
First, you need to download the datasets from their original source (If you have already downloaded, please ignore this step):
* [SSV2](https://20bn.com/datasets/something-something#download)
* [Kinetics](https://github.com/Showmax/kinetics-downloader)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
* [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Then, prepare data according to the [splits](https://github.com/D-XH/CLIP-MEI/tree/main/splits) we provide.
Alternatively, you can directly utilize the preprocessed [dataset](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main) we provide.

# Train and test
Modify the path of the config file within the `train_test.sh` file. Then the codebase can be run by:
```shell
sh train_test.sh
```
You need to alter the values of certain fields within the config file before.
* `DATA_DIR`: The root path of datasets
* `ONLY_TEST`: Set `true` for testing, and set `false` for training.
* `WAY`: The number of class per episode.
* `SHOT`: The number of videos per class in support set.
* `QUERY_PER_CLASS`: The number of videos per class in querry set.
* `TEST_MODEL_PATH`: The path of checkpoint for test (`ONLY_TEST: true`).

# Checkpoints
We provide the checkpoint for all the results presented in our paper.
* [HMDB51](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main/checkpoints/hmdb)
* [UCF101](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main/checkpoints/ucf)
* [SSv2-otam](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main/checkpoints/ssv2-otam)
* [Kinetics](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main/checkpoints/k100)
* [Ablations](https://openxlab.org.cn/datasets/DENG-H/FSAR/tree/main/checkpoints/xr)

# Citation
If you find this code useful, please cite our paper.
```
@article{deng2025clip,
  title={CLIP-MEI: Exploit more effective information for few-shot action recognition},
  author={Deng, XuanHan and Yang, WenZhu and Zhao, XinBo and Zhou, Tong and Deng, Xin},
  journal={Knowledge-Based Systems},
  pages={113965},
  year={2025},
  publisher={Elsevier}
}
```
