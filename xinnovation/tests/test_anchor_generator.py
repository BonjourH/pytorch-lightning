import pytest
from xinnovation.src.core.registry import ANCHOR_GENERATOR
from src.components.lightning_module.detectors.plugins import Anchor3DGenerator


@pytest.mark.parametrize("generator_config", [
    {
        "type": "Anchor3DGenerator",
        "front_type": "div_x",
        "back_type": "div_x",
        "front_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0},
        "back_params": {"alpha": 0.6, "beta": 9.0, "order": 2.0},
        "front_min_spacing": 2.5,
        "front_max_distance": 200.0,
        "back_min_spacing": 2.5,
        "back_max_distance": 100.0,
        "left_y_max": 3.75 * 3,
        "right_y_max": 3.75 * 3,
        "y_interval": 3.75,
        "z_value": 0.2,
        "anchor_size": (5.0, 2.0, 1.5)
    }
])        
def test_anchor_generator(generator_config):
    # Create a basic generator
    generator = ANCHOR_GENERATOR.build(generator_config)
    
    # Test the save_bev_anchor_fig function
    generator.save_bev_anchor_fig(output_dir="/tmp")
