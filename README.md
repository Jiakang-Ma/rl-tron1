# RL-TRON1: Bipedal Robot Reinforcement Learning Locomotion

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

A reinforcement learning framework for training bipedal robot locomotion using [Isaac Lab](https://github.com/isaac-sim/IsaacLab). This project enables training and deploying locomotion policies for the [LimX Dynamics TRON1](https://www.limxdynamics.com/en/tron1) point-foot bipedal robot.

## Features

- 🤖 **Bipedal Locomotion Training** - Train point-foot robots to walk on flat, rough terrain, and stairs
- 🧠 **Teacher-Student Learning** - Concurrent teacher-student RL framework with MLP encoder
- 🎮 **Sim2Sim Transfer** - Deploy trained policies to MuJoCo simulation
- 🦾 **Sim2Real Ready** - Export policies for real robot deployment
- ⚡ **GPU-Accelerated** - Parallel training with thousands of environments

## Demo

<table>
<tr>
<td align="center" width="50%">

**Isaac Lab Training**

![Isaac Lab Training](./media/play_isaaclab.gif)

</td>
<td align="center" width="50%">

**MuJoCo Sim2Sim Deployment**

![MuJoCo Deployment](./media/play_mujoco.gif)

</td>
</tr>
</table>

## Project Structure

```
rl-tron1/
├── exts/bipedal_locomotion/     # Isaac Lab extension for bipedal locomotion
│   └── bipedal_locomotion/
│       ├── assets/              # Robot configuration files
│       ├── tasks/locomotion/    # Task definitions and rewards
│       └── utils/               # Utility functions and wrappers
├── scripts/rsl_rl/              # Training and evaluation scripts
│   ├── train.py                 # Training script
│   └── play.py                  # Policy evaluation script
├── rsl_rl/                      # Modified RSL-RL library with MLP support
├── robot_desc/                  # Robot URDF/USD descriptions
├── motion_data/                 # Reference motion data for imitation learning
├── exported_models/             # Trained policy exports (ONNX, JIT)
├── tron1-rl-deploy-python/      # Python deployment package for sim2sim
└── pointfoot-mujoco-sim/        # MuJoCo simulation environment
```

## Installation

### Prerequisites

- Ubuntu 20.04/22.04
- NVIDIA GPU with driver 525+
- Python 3.10
- [Isaac Sim 4.5.0](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)

### Step 1: Install Isaac Lab

**Option A: One-Click Installation (Recommended)**

```bash
wget -O install_isaaclab.sh https://docs.robotsfan.com/install_isaaclab.sh && bash install_isaaclab.sh
```
> Select Isaac Sim version 2.1.0 when prompted. Thanks to [@fan-ziqi](https://github.com/fan-ziqi) for this installation script.

**Option B: Official Installation**

Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html).

### Step 2: Clone Repository

```bash
# Clone outside the IsaacLab directory
git clone https://github.com/Jiakang-Ma/rl-tron1.git
cd rl-tron1
```

### Step 3: Install Dependencies

```bash
conda activate isaaclab

# Install bipedal locomotion extension
python -m pip install -e exts/bipedal_locomotion

# Install modified RSL-RL with MLP support
cd rsl_rl && python -m pip install -e . && cd ..
```

### Step 4: IDE Setup (Optional)

Update `.vscode/settings.json` with your Isaac Lab and Python paths for code navigation support.

## Usage

### Training

```bash
python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Flat-v0 --headless
```

**Training Options:**
| Argument | Description |
|----------|-------------|
| `--headless` | Run without rendering |
| `--num_envs` | Number of parallel environments (default: 4096) |
| `--max_iterations` | Maximum training iterations |
| `--save_interval` | Checkpoint save interval |
| `--seed` | Random seed |

### Evaluation

```bash
python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint
```

**Evaluation Options:**
| Argument | Description |
|----------|-------------|
| `--num_envs` | Number of environments |
| `--headless` | Run without rendering |
| `--checkpoint_path` | Path to trained checkpoint |

## Sim2Sim Deployment (MuJoCo)

After training, deploy your policy to MuJoCo for validation:

1. Export your trained policy to ONNX format (automatic after `play.py`)
2. Navigate to the MuJoCo simulation:
   ```bash
   cd pointfoot-mujoco-sim
   ```
3. Replace `policy.onnx` and `encoder.onnx` with your trained models
4. Run simulation:
   ```bash
   python simulator.py
   ```

For detailed instructions, see:
- [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python)
- [pointfoot-mujoco-sim](https://github.com/limxdynamics/pointfoot-mujoco-sim)

## Sim2Real Deployment

The policies are trained using PPO within an asymmetric actor-critic framework, with actions determined by history observations latent and proprioceptive observation.

**Framework Reference:** Inspired by [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://doi.org/10.1109/LRA.2024.3457379) (H. Wang, H. Luo, W. Zhang, and H. Chen, 2024).

For real robot deployment, refer to the [official LimX documentation](https://support.limxdynamics.com/docs/tron-1-sdk/rl-training-results-deployment) (Sections 8.1-8.2).

## Acknowledgements

This project builds upon these open-source projects:
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab)
- [tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab)

## Contributors

- Jiakang Ma
- Xihua Zhang
- Hongwei Xiong 
- Bobin Wang
- Wen
- Haoxiang Luo
- Junde Guo

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.