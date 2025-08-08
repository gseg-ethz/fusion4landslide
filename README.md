<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images</h1>
  <p align="center">
    <a href="https://github.com/zhaoyiww/fusion4landslide"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/zhaoyiww/fusion4landslide"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://arxiv.org/abs/2506.16265"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/zhaoyiww/fusion4landslide/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>  

  <p align="center">
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/zhaoyi-wang.html"><strong>Zhaoyi Wang</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/jemil-avers-butt.html"><strong>Jemil Avers Butt</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people.html"><strong>Shengyu Huang</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/tomislav-medic.html"><strong>Tomislav Medic</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html"><strong>Andreas Wieser</strong></a>
  </p>
    <p align="center">
      <a href="https://gseg.igp.ethz.ch/people/people-group.html"><strong>Geosensors and Engineering Geodesy (GSEG) Lab</strong></a>, 
      <a href="https://igp.ethz.ch/"><strong>Institute of Geodesy and Photogrammetry (IGP)</strong></a>, 
      <a href="https://ethz.ch/en.html"><strong>ETH Zürich</strong></a>
    </p>

---

This repository contains official implementations of our series of work in TLS-based landslide monitoring. ⭐ Star us if you find it useful!

- 📷 [RGB-Guided Dense 3D Displacement Estimation in TLS-Based Geomonitoring](https://www.research-collection.ethz.ch/handle/20.500.11850/731656) *(ISPRS Geospatial Week, 2025)*

- 🏔️ [Dense 3D Displacement Estimation via Fusion of TLS Point Clouds and Embedded RGB Images](https://arxiv.org/abs/2506.16265) *(ArXiv, 2025)*

The `main` branch contains the official implementations of both works and baselines ([Piecewise ICP](https://fig.net/resources/proceedings/2016/2016_03_jisdm_pdf/nonreviewed/JISDM_2016_submission_97.pdf), [F2S3](https://link.springer.com/article/10.1007/s10346-021-01761-y)). For the code specific to the RGB-Guided approach, switch to the [`rgb-guided-only`](https://github.com/zhaoyiww/fusion4landslide/tree/rgb-guided-only) branch.

---

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>📚 Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#example-data">Example data</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#todo-list">TODO list</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## 🛠️ Installation <a name="installation"></a>

Clone the repository and set up the environment:

```bash
git clone --recursive git@github.com:zhaoyiww/fusion4landslide.git
cd fusion4landslide
bash install.sh
```

## 📁 Example data <a name="example-data"></a>

We provide our collected **Rockfall Simulator** dataset: [📦 Download from Hugging Face](https://huggingface.co/datasets/zhaoyiww/Rockfall_Simulator/tree/main). This dataset is partially used in our [RGB-Guided paper](https://www.research-collection.ethz.ch/handle/20.500.11850/731656). Please refer to the paper for a detailed data description and feel free to explore the full dataset for your own research.

## 🚀 Run <a name="run"></a>
Before running, modify the corresponding config file in `/configs/test/[method]_[dataset].yaml`, and update any necessary parameters in `main_[method].py`. Then, run the main pipeline using:
```bash
python main_fusion.py          # run for TLS+RGB fusion method
python main_rgb_guided.py      # run for RGB-Guided method
python main_f2s3.py            # run for F2S3 baseline
python main_piecewise_icp.py   # run for Piecewise ICP baseline
```

## 📌 TODO list <a name="todo-list"></a>
- [x] Add sample data and RGB-Guided scripts.
- [ ] ⏳ Add baselines for quantitative and qualitative comparison.
- [ ] ⏳ Release full RGB-3D fusion pipeline upon paper acceptance.
- [ ] ⏳ Extend to photogrammetric 3D point clouds (SfM/MVS).
- [ ] ⏳ Provide Pythonic version of point cloud tiling.

## 🤝 Acknowledgements

We gratefully acknowledge the following open-source projects that contributed to this work:
- Superpoint segmentation ([original](https://github.com/drprojects/superpoint_transformer) · [customized](https://github.com/zhaoyiww/superpoint_transformer)): Used for generating hierarchical patches in our pipeline.
- Supervoxel segmentatition ([original](https://github.com/yblin/Supervoxel-for-3D-point-clouds) · [customized](https://github.com/gseg-ethz/supervoxel?tab=readme-ov-file)): Incorporated for single patch generation.
- Efficient LoFTR ([original](https://github.com/zju3dv/EfficientLoFTR) · [customized](https://github.com/zhaoyiww/EfficientLoFTR)]): Used for semi-dense image pixel matching in our RGB-Guided method, and as the image-matching module in our fusion approach.
- RoMA ([original](https://github.com/Parskatt/RoMa) · [customized](https://github.com/zhaoyiww/RoMa)): Integrated for dense image pixel matching.

## 🔗 Relevant projects
- [py4dgeo](https://github.com/3dgeo-heidelberg/py4dgeo): Implements M3C2 and its variants.
- [iecepy4D](https://github.com/franioli/icepy4d): A Python package for image-based glacier monitoring.
- [Piecewise-ICP](https://github.com/yihui4d/Piecewise-ICP?tab=readme-ov-file): A 4D point cloud registration method.

## 🤗 Citation <a name="citation"></a>
If our work helps your research, please consider citing:

```bibtex
@preprint{wang2025fusion4landslide,
  title={Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images},
  author={Wang, Zhaoyi and Butt, Jemil Avers and Huang, Shengyu and Medic, Tomislav and Wieser, Andreas},
  journal={arXiv preprint},
  year={2025},}
```

```bibtex
@article{wang2025RGB4landslide,
  title={An approach for rgb-guided dense 3d displacement estimation in tls-based geomonitoring},
  author={Wang, Zhaoyi and Butt, Jemil Avers and Huang, Shengyu and Meyer, Nicholas and Medi{\'c}, Tomislav and Wieser, Andreas},
  journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  pages={953--960},
  year={2025},
}
```
