# CLIP-MEI: Exploit more effective information for few-shot action recognition
Official Pytorch Implementation of CLIP-MEI
> Exploit more effective information for few-shot action recognition (CLIP-MEI)
> Xuanhan Deng

> Abstract:

This code is based on [TRX](https://github.com/tobyperrett/trx) codebase.

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
...
```
