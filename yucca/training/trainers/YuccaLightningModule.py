import lightning as L
import torch
import yuccalib
import wandb
import yucca
from batchgenerators.utilities.file_and_folder_operations import join
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice
from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
from yuccalib.utils.files_and_folders import recursive_find_python_class
from yuccalib.utils.kwargs import filter_kwargs
from typing import Literal


class YuccaLightningModule(L.LightningModule):
    """
    The YuccaLightningModule class is an implementation of the PyTorch Lightning module designed for the Yucca project.
    It extends the LightningModule class and encapsulates the neural network model, loss functions, and optimization logic.
    This class is responsible for handling training, validation, and inference steps within the Yucca machine learning pipeline.


    """

    def __init__(
        self,
        configurator=YuccaConfigurator,
        learning_rate: float = 1e-3,
        loss_fn: str = "DiceCE",
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
        momentum: float = 0.9,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        sliding_window_overlap: float = 0.5,
        stage: Literal["fit", "test", "predict"] = "fit",
        step_logging: bool = False,
        test_time_augmentation: bool = False,
    ):
        super().__init__()
        # Extract parameters from the configurator
        self.num_classes = configurator.num_classes
        self.num_modalities = configurator.num_modalities
        self.outpath = configurator.outpath
        self.plans = configurator.plans
        self.plans_path = configurator.plans_path
        self.model_name = configurator.model_name
        self.model_dimensions = configurator.model_dimensions
        self.patch_size = configurator.patch_size

        # Loss, optimizer and scheduler parameters
        self.lr = learning_rate
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = "DiceCE"
        self.momentum = momentum
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        # Evaluation and logging
        self.step_logging = step_logging
        self.train_metrics = MetricCollection({"train_dice": Dice(num_classes=self.num_classes, ignore_index=0)})
        self.val_metrics = MetricCollection({"val_dice": Dice(num_classes=self.num_classes, ignore_index=0)})

        # Inference
        self.sliding_window_overlap = sliding_window_overlap
        self.test_time_augmentation = test_time_augmentation

        # If we are training we save params and then start training
        # Do not overwrite parameters during inference.
        self.save_hyperparameters()
        self.load_model()

    def load_model(self):
        print(f"Loading Model: {self.model_dimensions} {self.model_name}")
        self.model = recursive_find_python_class(
            folder=[join(yuccalib.__path__[0], "network_architectures")],
            class_name=self.model_name,
            current_module="yuccalib.network_architectures",
        )
        if self.model_dimensions == "3D":
            conv_op = torch.nn.Conv3d
            norm_op = torch.nn.InstanceNorm3d
        else:
            conv_op = torch.nn.Conv2d
            norm_op = torch.nn.BatchNorm2d

        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # UNetR
            "patch_size": self.patch_size,
            # MedNeXt
            "checkpoint_style": None,
        }
        model_kwargs = filter_kwargs(self.model, model_kwargs)

        self.model = self.model(**model_kwargs)

    def forward(self, inputs):
        return self.model(inputs)

    def teardown(self, stage: str):
        wandb.finish()

    def training_step(self, batch, batch_idx):
        inputs, target = batch["image"], batch["seg"]
        output = self(inputs)
        loss = self.loss_fn(output, target)
        metrics = self.train_metrics(output, target)
        self.log_dict({"train_loss": loss} | metrics, on_step=self.step_logging, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch["image"], batch["seg"]
        output = self(inputs)
        loss = self.loss_fn(output, target)
        metrics = self.val_metrics(output, target)
        self.log_dict({"val_loss": loss} | metrics, on_step=self.step_logging, on_epoch=True, prog_bar=False, logger=True)

    def on_predict_start(self):
        preprocessor_class = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "preprocessing")],
            class_name=self.plans["preprocessor"],
            current_module="yucca.preprocessing",
        )
        self.preprocessor = preprocessor_class(join(self.outpath, "hparams.yaml"))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        case, case_id = batch

        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(case, self.patch_size)

        logits = self.model.predict(
            mode=self.model_dimensions,
            data=case_preprocessed,
            patch_size=self.patch_size,
            overlap=self.sliding_window_overlap,
            mirror=self.test_time_augmentation,
        )

        logits = self.preprocessor.reverse_preprocessing(logits, case_properties)
        return {"logits": logits, "properties": case_properties, "case_id": case_id[0]}

    def configure_optimizers(self):
        # Initialize and configure the loss(es) here.
        # loss_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        self.loss_fn = recursive_find_python_class(
            folder=[join(yuccalib.__path__[0], "loss_and_optim")],
            class_name=self.loss_fn,
            current_module="yuccalib.loss_and_optim",
        )
        loss_kwargs = {
            # DCE
            "soft_dice_kwargs": {"apply_softmax": True},
        }

        loss_kwargs = filter_kwargs(self.loss_fn, loss_kwargs)

        self.loss_fn = self.loss_fn(**loss_kwargs)

        # Initialize and configure the optimizer(s) here.
        # optim_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        optim_kwargs = {
            # all
            "lr": self.lr,
            # SGD
            "momentum": self.momentum,
            "eps": 1e-4,
            "weight_decay": 3e-5,
        }

        optim_kwargs = filter_kwargs(self.optim, optim_kwargs)

        self.optim = self.optim(self.model.parameters(), **optim_kwargs)

        # Initialize and configure LR scheduler(s) here
        # lr_scheduler_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        lr_scheduler_kwargs = {
            # Cosine Annealing
            "T_max": self.trainer.max_epochs,
            "eta_min": 1e-9,
        }

        lr_scheduler_kwargs = filter_kwargs(self.lr_scheduler, lr_scheduler_kwargs)

        self.lr_scheduler = self.lr_scheduler(self.optim, **lr_scheduler_kwargs)

        # Finally return the optimizer and scheduler - the loss is not returned.
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}


if __name__ == "__main__":
    from yucca.training.trainers.YuccaLightningManager import YuccaLightningManager

    path = None
    Manager = YuccaLightningManager(
        task="Task001_OASIS",
        limit_train_batches=250,
        limit_val_batches=50,
        max_epochs=5,
        ckpt_path=path,
    )
    # Manager.initialize()
    # Manager.run_training()
