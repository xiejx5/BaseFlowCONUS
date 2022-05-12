import hydra
import logging
from hydra.utils import instantiate
from src.utils import consistent_split
from src.trainer.trainer import Trainer


@hydra.main(config_path='configs', config_name='config')
def main(cfg):
    logger = logging.getLogger('trainer')

    # dynamic or static dynamic dataset
    dataset = instantiate(cfg.dataset)
    # setup data_loader instances
    datasplit = instantiate(cfg.datasplit, dataset)
    # set a consistent stratify split if training on all ecoregions
    consistent_split(cfg, dataset, datasplit)

    n, accumulation = 0, 0
    for train_loader, test_loader in datasplit:
        # build model and print its architecture
        model = instantiate(cfg.model)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        logger.info(model)
        logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

        # get function handles of loss and metrics
        criterion = instantiate(cfg.loss, dataset)
        metrics = [instantiate(met) for met in cfg.trainer['metrics']]

        # build optimizer and learning rate scheduler
        optimizer = instantiate(cfg.optimizer, model.parameters())
        lr_scheduler = instantiate(cfg.trainer.lr_scheduler, optimizer)

        # train and accumulate metric
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=cfg,
                          data_loader=train_loader,
                          valid_data_loader=test_loader,
                          lr_scheduler=lr_scheduler,
                          round=n)
        accumulation += trainer.train()
        n += 1

    return accumulation / n


if __name__ == '__main__':
    main()
