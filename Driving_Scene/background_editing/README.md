Background Editing

1. Download Dataset & Pretrained Model
Download sample dataset from here [here](https://drive.google.com/file/d/1YbDlD0X7YOwhQuvqyFnnVOfFPXfeu9q4/view?usp=sharing) and extract it under folder dataset
Download model from [here](https://drive.google.com/file/d/1r38vaV7lb4tFVyq6n3twoDwa5qhqOlti/view?usp=sharing)
2. Run the inference 
python src/inference.py --output_dir "outputs/waymo_snowy"  --dataset waymo --weather_type snowy


Particle Construction 
#### 1. Download Dataset 
Please follow OmniRe's instruction [here](https://github.com/ziyc/drivestudio?tab=readme-ov-file) to download dataset
#### 2. 3D Scene Reconstruction
Run the scripts
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep


#### 3. Render with Weather Effects

python src/inference.py --output_dir "outputs/dataset_name"  --dataset dataset_name --weather_type snowy
