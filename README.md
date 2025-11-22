<h2 align="center">
  🌦️ WeatherEdit: Controllable Weather Editing with 4D Gaussian Field
</h2>
<p align="center">
  <em>
    <a href="https://jumponthemoon.github.io/">Chenghao Qian</a><sup>1,*</sup>,
    <a href="https://scholar.google.com/citations?user=uBjSytAAAAAJ&hl=en">Wenjing Li</a><sup>1,†</sup>,
    <a href="https://example.com/guo">Yuhu Guo</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=55G7VxoAAAAJ&hl=en">Gustav Markkula</a><sup>1</sup>
  </em>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/4483e88e-4552-4245-9c29-2c84abf78788" alt="image" width="390">
</p>



<p align="center">
  <a href="https://arxiv.org/abs/2505.20471">
    <img src="https://img.shields.io/badge/arXiv-article-red" alt="arXiv">
  </a>
  <a href="https://jumponthemoon.github.io/w-edit">
    <img src="https://img.shields.io/badge/Project-link-blue" alt="Project">
  </a>
</p>






<p align="center">
  A <strong>Controllable</strong>, <strong>Scalable</strong> and <strong>Efficient</strong> Framework for <strong>Realistic Weather Editing.</strong>
</p>
<div align="center">
  <img src="https://github.com/user-attachments/assets/b2b46e74-010d-4d80-a153-e4561668cc0f" width="800"/>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/e5b74e41-d25f-4a25-af42-99a95bcdb04e" width="800"/>
</div>



---

- 🎨 Flexible control over **weather types** (snow, fog, rain)
- 🌡️ Precise **weather severity** adjustment (light, moderate, heavy)
- 🖼️ **Global consistency** for multi-view driving scenes (temporal, spatial)

---

## 💡 The Quickiest Start
You can use our provided pretrained model for the easiest start. If you would like to try the whole pipeline, please go to the subfolder for training instructions
### A. General Weather (`General Scene/`)
Please first configure the environment with conda:
```bash
git clone https://github.com/Jumponthemoon/WeatherEdit.git
cd General_Scene
conda env create --file environment.yml
conda activate gaussian_splatting
```
We provide a complete pipeline to train and render Gaussian scenes with integrated weather effects. 

#### 1. Train Your Scene
Download the Dataset Pt.1 from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and put the `garden` under `data` folder, then run:
```bash
python train.py -s path/to/data/
```
After training, the model will be saved under `output` folder
#### 2. Render with Weather Effects

```bash
python render.py -m path/to/model --weather snow --fps 10
```

🔥 **Plug into your GS-based code?**  👉 Check it out [here](https://github.com/Jumponthemoon/WeatherEdit/tree/main/General_Scene)

---

### B. Driving Scene Editing (`Driving_Scene/`)
Please first clone the repo and configure the environment with conda:
```bash
git clone https://github.com/Jumponthemoon/WeatherEdit.git
cd Driving_Scene

conda create -n weatheredit python=3.9 -y
conda activate weatheredit

pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

cd third_party/smplx/
pip install -e .
cd ../..
```
* Note： if you encounter error `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`, please follow [this](https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2425265522) instruction.

#### 1. Download sample dataset & pretrained model
```bash
cd particle_construction
```
Download [sample dataset](https://drive.google.com/file/d/18qwNg_VVcwiyliLW1eDq488lRe8mdnuX/view?usp=sharing) and [pretrained model](https://drive.google.com/file/d/1vXz_-tPkwEU61jFrke9Io044An1j4Bv4/view?usp=sharing), then place them in the `data` and the `output` directory separately.


#### 2. Render with Weather Effects
Run the script to generate rainy weather in pandaset:
```bash
export PYTHONPATH=$(pwd)
python tools/gen_particle.py --resume_from ./output/pandaset/44/checkpoint_final.pth --weather rainy
```
The rendered video will be saved under `./output/pandaset/44/video_eval` folder

> ⭐ **If you like our work or find it useful, please give us a star or cite below. Thanks!**

---



## 📌 Citation

```bibtex
@article{qian2025wedit,
      title={WeatherEdit: Controllable Weather Editing with 4D Gaussian Field}, 
      author={Chenghao Qian and Wenjing Li and Yuhu Guo and Gustav Markkula},
      year={2025},
      eprint={2505.20471},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20471},}  
```

---

## 📬 Contact

For questions, suggestions, or collaborations:

- 📧 tscq@leeds.ac.uk
---

Thanks for your interest in WeatherEdit! We hope it helps bring new life to your 3D scenes 🌧️🌨️🌫️
