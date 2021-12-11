import sys
sys.path = ['', '..'] + sys.path

import hydra
from application.app import create_app
from hydra.utils import to_absolute_path
from omegaconf.dictconfig import DictConfig


@hydra.main(config_path="conf", config_name="flask")
def main(cfg: DictConfig):
    cfg['model']['checkpoint'] = to_absolute_path(cfg['model']['checkpoint'])
    app = create_app(cfg)
    app.run()

if __name__ == "__main__":
    main()