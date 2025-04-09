<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">An Approach for RGB-Guided Dense 3D Displacement Estimation in TLS-Based Geomonitoring</h1>
  <p align="center">
    <a href="https://github.com/zhaoyiww/fusion4landslide"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/zhaoyiww/fusion4landslide"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="/assets/Camera_Ready_Version.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/zhaoyiww/fusion4landslide/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>  

  <p align="center">
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/zhaoyi-wang.html"><strong>Zhaoyi Wang</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/jemil-avers-butt.html"><strong>Jemil Avers Butt</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people.html"><strong>Shengyu Huang</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/nicholas-meyer.html"><strong>Nicholas Meyer</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/tomislav-medic.html"><strong>Tomislav Medic</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html"><strong>Andreas Wieser</strong></a>
  </p>
  <p align="center"><a href="https://ethz.ch/en.html"><strong>ETH Zürich</strong></a>

  <div align="center"></div>
</p>

This is the official PyTorch implementation of our paper accepted at the ISPRS Laser Scanning Workshop 2025 (Dubai):

📄 [Paper (Camera Ready)](assets/Camera_Ready_Version.pdf)

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#to-do">To Do</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>



## 🛠️ Installation

Clone the repository and set up the environment:

```bash
git clone --recursive git@github.com:zhaoyiww/fusion4landslide.git
cd fusion4landslide
sh install.sh
```

## 🚀 Run
Run the main pipeline:
```bash
python main_rgb_guided.py
```

## 📌 To-Do
- Add demo scripts and sample data
- Add baselines for quantitative and qualitative comparison
- Release the RGB-3D fusion pipeline upon paper submission

## 📖 Citation
If you find this work useful, please consider citing:

```bash
@inproceedings{Wang2025RGB4landslide,
  title={An Approach for RGB-Guided Dense 3D Displacement Estimation in TLS-Based Geomonitoring},
  author={Wang, Z., Butt, J., Huang, S., Meyer, N., Medić, T., and Wieser, A.},
  journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year={2025}
```
