import torch

import colossalai
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger

import wandb

from lamda_pytorch.config.config import CFG
from lamda_pytorch.build_dataloader import build_dataloaders
from lamda_pytorch.lamda_pytorch import lamda_model
from lamda_pytorch.utils.utils import LaMDA_Loss, AutoregressiveWrapper

from transformers import AutoTokenizer

from accelerate import Accelerator, DistributedType


EPOCHS = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-2

gradient_accumulation = 1
clip_grad_norm = 0.0

def LaMDA_Trainer(cfg: CFG):
    assert torch.cuda.is_available()
    disable_existing_loggers()

    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer',
        action='store_true',
        help='whether to use trainer'
    )

    args = parser.parse_args()

    accelerator = Accelerator()
    #
    # if cfg.use_zero == True:
    #     pass
    # else:
    #     colossalai.launch_from_torch(
    #         config='./lamda_pytorch/config/colossal_config.py',
    #         seed=cfg.seed
    #     )

    # assert hasattr(gpc.config, "EPOCHS"), "Please provide NUM_EPOCHS in your configuration"

    # Colossal logger
    logger = get_dist_logger()
    logger.info("Initialized environment", ranks=[0])

    # LaMDA model
    model = lamda_model()
    model = AutoregressiveWrapper(model)

    # setup dataloaders
    if cfg.use_huggingface == True:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        train_dataloader, eval_dataloader = build_dataloaders(cfg, tokenizer)

    # loss function
    loss_fn = LaMDA_Loss()

    # optimizer function

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # initialze model, optimizer, criterion, and data loaders

    # engine, train_dataloader, _, _ = colossalai.initialize(
    #     model,
    #     optimizer,
    #     loss_fn,
    #     train_dataloader=train_dataloader
    # )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    def batch_data_process_func(batch_data):
        data = batch_data["input_ids"]
        labels = batch_data["labels"]
        return data, labels

    # engine.schedule.batch_data_process_func = batch_data_process_func

    if cfg.use_wandb == True:

        # initialize Weights and Biases Logging
        wandb.init(project=cfg.project_name)

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(accelerator.device)
            inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()

            # engine.zero_grad()
            outputs = model(inputs)

            train_loss = loss_fn.forward(outputs, labels)
            print()
            wandb.log({"train_loss": train_loss})

            accelerator.backward(train_loss)
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"step": step})

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                inputs, labels = batch['input_ids'].cuda(), batch['labels'].cuda()

                with torch.no_grad():
                    outputs = model(inputs)
                    test_loss = loss_fn.forward(outputs, labels)
                    wandb.log({"test_loss": test_loss})

                # engine.backward(test_loss)
                # engine.step()

        wandb.alert(
            title='Training Complete',
            text="Training complete."
        )

    else:

        # Time session with ColossalAI
        timer = MultiTimer()

        # trainer
        trainer = Trainer(
            engine=engine,
            timer=timer,
            logger=logger
        )

        hook_list = [
            hooks.LogMetricByStepHook(),
            hooks.LossHook(),
            hooks.LogMetricByEpochHook(logger)
        ]

        trainer.fit(
            train_dataloader=train_dataloader,
            epochs=EPOCHS,
            hooks=hook_list,
            display_progress=True
        )


if __name__ == "__main__":
    cfg = CFG()

    LaMDA_Trainer(cfg)
