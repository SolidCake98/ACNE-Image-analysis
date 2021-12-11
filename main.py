from omegaconf.dictconfig import DictConfig
from LDL import factory
import hydra
from hydra.utils import to_absolute_path


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):

    cfg['dataset_local_train']['data_path'] = to_absolute_path(cfg['dataset_local_train']['data_path'])
    cfg['dataset_local_train']['data_file'] = to_absolute_path(cfg['dataset_local_train']['data_file'])

    cfg['dataset_local_test']['data_path'] = to_absolute_path(cfg['dataset_local_test']['data_path'])
    cfg['dataset_local_test']['data_file'] = to_absolute_path(cfg['dataset_local_test']['data_file'])

    trainer, model, train_loader, test_loader = factory.get_trainer(
        cfg['backbone']['name'],
        cfg['dataset_local_train'],
        cfg['dataset_local_test'],
        cfg['logger'],
        cfg['trainer']
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    train()