# Background Editing & Particle Construction Pipeline

## 🚀 Background Editing

Edit backgrounds with **snowy**, **rainy**, and **foggy** effects.

------------------------------------------------------------------------

### **1. Download Dataset & Pretrained Model**

-   **Sample Dataset**\
    👉 [Download](https://drive.google.com/file/d/1YbDlD0X7YOwhQuvqyFnnVOfFPXfeu9q4/view?usp=sharing) and extract into the `datasets/` folder.

-   **Pretrained Model**\
    👉 [Download](https://drive.google.com/file/d/1r38vaV7lb4tFVyq6n3twoDwa5qhqOlti/view?usp=sharing) and place it in the `ckpts/` folder.

------------------------------------------------------------------------

### **2. Run Inference**
Our default setup is for multi-view datasets.
But if you're working with just single-view and single-frame images, we've got you covered!
Please prepare your images and mask following the sample dataset structure with filename suffix `_0`, then use `--dataset custom` for inference.

#### **Multi-View & Multi-Frame**

``` bash
python src/inference.py     --output_dir "outputs/waymo_snowy"     --dataset waymo     --weather_type snowy
```

#### **Single-View & Single-Frame**

``` bash
python src/inference.py     --output_dir "outputs/custom_snowy"     --dataset custom     --weather_type snowy
```
------------------------------------------------------------------------

## ❄️ Particle Construction

Driving scene reconstruction and generate **dynamic 3D particles** (e.g., snow, rain, fog) in those scenes.

------------------------------------------------------------------------

### **1. Download Dataset**

Follow the dataset download instructions from [OmniRe](https://github.com/ziyc/drivestudio?tab=readme-ov-file)

------------------------------------------------------------------------

### **2. 3D Scene Reconstruction Training**

Replace the dataset images with your **edited** ones, then run:

``` bash
export PYTHONPATH=$(pwd)

start_timestep=0   # Starting frame index
end_timestep=-1    # -1 = use last frame
python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=pandaset/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

------------------------------------------------------------------------

### **3. Render with Dynamic Particles**

Different datasets require different particle settings. Specify the corresponding `--dataset` name, then run:

``` bash
python src/inference.py     --output_dir "outputs/dataset_name"     --dataset dataset_name     --weather_type snowy
```

If you want to adjust the severity or the results aren't satisfying, feel free to tune the parameters in `configs/particle_config.yaml`
