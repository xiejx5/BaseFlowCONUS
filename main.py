from hydra.experimental import compose, initialize_config_dir
import hydra
import os
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path='conf/', config_name='config.yaml')
def main(config):
    print(f"Current working directory : {os.getcwd()}")
    print(f"{config.work_dir}")
    print(f"{config.work_dir}")


initialize_config_dir(config_dir=os.path.join(os.getcwd(), 'conf'))
config = compose(config_name="config", return_hydra_config=True)
print(OmegaConf.to_yaml(config))
