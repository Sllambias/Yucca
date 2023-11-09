from yucca.training.trainers.YuccaManager import YuccaManager


class YuccaManager_ForegroundOnly(YuccaManager):
    def __init__(
        self,
        model,
        model_dimensions: str,
        task: str,
        folds: str | int,
        plan_id: str,
        starting_lr: float = None,
        loss_fn: str = None,
        momentum: float = None,
        continue_training: bool = False,
        checkpoint: str = None,
        finetune: bool = False,
        fast_training: bool = False,
    ):
        super().__init__(
            model,
            model_dimensions,
            task,
            folds,
            plan_id,
            starting_lr,
            loss_fn,
            momentum,
            continue_training,
            checkpoint,
            finetune,
            fast_training,
        )
        self.p_force_foreground = 1.0
