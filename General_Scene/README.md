### General Weather Scene Editing
Please first configure the environment with conda:
```bash
cd General_Scene
conda env create --file environment.yml
conda activate gaussian_splatting
```
We provide a complete pipeline to train and render Gaussian scenes with integrated weather effects. 

#### 1. Train Your Scene
Download the sample garden scene [here](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and put the garden under `data` folder, then run:
```bash
python train.py -s path/to/data/
```
After training, the model will be saved under `output` folder
#### 2. Render with Weather Effects

```bash
python render.py -m path/to/model --weather snow --fps 10
```

---

### 🔌 Plug-and-Play Weather Modules

You can easily integrate dynamic weather into any Gaussian scene:

#### Step 1. Configure Weather Settings

Edit `weather_config.json`:

```json
{
  "weather_type": "snow",
  "density": 1500,
  "velocity": [0.0, -0.4, 0.0],
  "scale": 0.04
}
```

#### Step 2. Load Weather Particles

```python
from scene.gaussian_particle import GaussianParticle
def load_particle_config(gaussians, weather_type="snow", config_path=None):
    if not os.path.exists(config_path):
        print("No weather config found.")
        return None
    with open(config_path, "r") as f:
        weather_config = json.load(f)
    if weather_type not in weather_config:
        print(f"Warning: weather type '{weather_type}' not found in config.")
        return None
    return GaussianParticle(
        config=weather_config[weather_type],
        scene_extent=7,
        sh_degree=gaussians.max_sh_degree
    )

particle = load_particle_config(gaussians, weather_type="snow",config_path='weather_config.json')
particle_gaussians = particle.get_static_gaussians()
```

#### Step 3. Fuse Scene and Weather Gaussians

```python
means3D   = torch.cat((means3D, pg['positions']), dim=0)
means2D   = torch.cat((means2D, torch.zeros_like(pg['positions'])), dim=0)
opacity   = torch.cat((opacity, pg['opacity']), dim=0)
scales    = torch.cat((scales, pg['scaling']), dim=0)
rotations = torch.cat((rotations, pg['rotation']), dim=0)
```

#### Step 4. Render and Update Particles

```python
rendering = render(interp_cam, gaussians, pipeline, background,
                   use_trained_exp=train_test_exp,
                   separate_sh=separate_sh,
                   pg=particle_gaussians)["render"]

if particle:
    particle.update_positions(delta_time)
```

This modular design enables **seamless weather injection** without altering core rendering or training logic.

---
