from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)

#############################
# Rough Terrain Configuration
#############################

# Blind rough terrain configuration - for training without vision sensors
BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,                        # Random seed for reproducibility
    size=(8.0, 8.0),               # Each terrain tile size 8x8 meters
    border_width=20.0,              # Border width
    num_rows=10,                    # Number of terrain rows
    num_cols=16,                    # Number of terrain columns
    horizontal_scale=0.1,           # Horizontal resolution
    vertical_scale=0.005,           # Vertical resolution
    slope_threshold=0.75,           # Slope threshold
    use_cache=True,                 # Use cache for faster generation
   
    # Sub-terrain configurations - define different types of terrain
    sub_terrains={
        # Flat terrain (25% proportion)
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        
        # Wave terrain (25% proportion)
        "waves": HfWaveTerrainCfg(
            proportion=0.25, 
            amplitude_range=(0.01, 0.06),      # Wave amplitude range [m]
            num_waves=10,                      # Number of waves
            border_width=0.25                  # Border width
        ),
        
        # Random grid terrain (25% proportion)
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, 
            grid_width=0.15,                   # Grid width
            grid_height_range=(0.01, 0.04),    # Grid height range [m]
            platform_width=2.0                 # Platform width
        ),
        
        # Random rough terrain (25% proportion)
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.25, 
            noise_range=(0.01, 0.06),          # Noise height range [m]
            noise_step=0.01,                   # Noise step
            border_width=0.25                  # Border width
        ),
    },
    
    curriculum=True,                    # Enable curriculum learning
    difficulty_range=(0.0, 1.0),       # Difficulty range 0-1
)

# Blind rough terrain play configuration - for policy evaluation
BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,                         # Reduced rows for testing
    num_cols=4,                         # Reduced columns for testing
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    
    sub_terrains={
        # Only keep three terrain types
        "waves": HfWaveTerrainCfg(
            proportion=0.33,                # 33% proportion
            amplitude_range=(0.01, 0.06), 
            num_waves=10, 
            border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2,                 # 20% proportion
            grid_width=0.33,                # Larger grid
            grid_height_range=(0.01, 0.04), 
            platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.34,                # 34% proportion
            noise_range=(0.01, 0.06), 
            noise_step=0.01, 
            border_width=0.25
        ),
    },
    
    curriculum=False,                   # No curriculum for testing
    difficulty_range=(1.0, 1.0),       # Fixed maximum difficulty
)


##################################
# Hard Rough Terrain Configuration
##################################

BLIND_HARD_ROUGH_TERRAINS_CFG = BLIND_ROUGH_TERRAINS_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.02

BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG = BLIND_ROUGH_TERRAINS_PLAY_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_step = 0.02

##############################
# Stairs Terrain Configuration
##############################

# Stairs terrain training configuration - for training stair climbing ability
STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),                  # Larger terrain tiles for stairs
    border_width=20.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    
    sub_terrains={
        # Pyramid stairs (40% proportion)
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),    # Step height range 5-20cm
            step_width=0.3,                    # Step width 30cm
            platform_width=3.0,                # Platform width 3m
            border_width=1.0,                  # Border width
            holes=False,                       # No holes
        ),
        
        # Inverted pyramid stairs (40% proportion)
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),    # Descending steps
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        
        # Pyramid slope (10% proportion)
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.4),            # Slope range 0-40%
            platform_width=2.0, 
            border_width=0.25
        ),
        
        # Inverted pyramid slope (10% proportion)
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, 
            slope_range=(0.0, 0.4), 
            platform_width=2.0, 
            border_width=0.25
        ),
    },
    
    curriculum=True,                        # Enable curriculum learning
    difficulty_range=(0.0, 1.0),
)

STAIRS_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)
