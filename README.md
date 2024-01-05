# AVFAS

Official Pytorch implementation of AVFAS.

Our paper is accpeted to ACM MM 2023:

[Multi-Modal and Multi-Scale Temporal Fusion Architecture Search for Audio-Visual Video Parsing](https://dl.acm.org/doi/10.1145/3581783.3611947)

## Environment

Setup environment:

```shell
pip install -r requirements.txt
```

## Data preparation

### Look Listen and Parse

Data can be downloaded from [Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing](https://github.com/YapengTian/AVVP-ECCV20/tree/master).


## Training and Evaluation

For searching, set the searchable architecture parameters and run: 

```shell
sh scripts/search.sh
```

For retraining the found genotype, set the `search_dir` and run: 

```shell
sh scripts/search.sh
```

For evaluating the model, set the the `result_dir` and run: 

```shell
sh scripts/eval.sh
```

## Checkpoints

The checkpoints and training log can be downloaded [here](https://drive.google.com/file/d/13Hkj-7uXNAgK9mwoc5rfY8iiRo7oTXGP/view?usp=sharing).

## Citation

If you find our work useful, please cite our paper:

```
@inproceedings{zhang2023avfas,
author = {Zhang, Jiayi and Li, Weixin},
title = {Multi-Modal and Multi-Scale Temporal Fusion Architecture Search for Audio-Visual Video Parsing},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611947},
doi = {10.1145/3581783.3611947},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {3328–3336},
numpages = {9},
location = {<conf-loc>, <city>Ottawa ON</city>, <country>Canada</country>, </conf-loc>},
series = {MM '23}
}
```

## Acknowledgement

We thank [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20/tree/master) for their great codebase.