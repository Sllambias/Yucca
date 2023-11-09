import argparse
import yucca
from yucca.paths import yucca_models
from yucca.utils.task_ids import maybe_get_task_from_task_id
from yuccalib.utils.files_and_folders import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def main():
    parser = argparse.ArgumentParser()

    # Required Arguments #
    parser.add_argument(
        "-t",
        "--target",
        help="Name of the target task to train on. "
        "The data should already be preprocessed using yucca_preprocess"
        "Argument should be of format: TaskXXX_MYTASK",
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Name of the source task with pretrained models. "
        "The data should already be preprocessed using yucca_preprocess"
        "Argument should be of format: TaskXXX_MYTASK",
    )

    # Optional arguments with default values #
    parser.add_argument(
        "-m",
        help="Model Architecture. Should be one of MultiResUNet or UNet"
        " Note that this is case sensitive. "
        "Defaults to the standard UNet.",
        default="UNet",
    )
    parser.add_argument(
        "-d",
        help="Dimensionality of the Model. Can be 3D or 2D. "
        "Defaults to 3D. Note that this will always be 2D if ensemble is enabled.",
        default="3D",
    )
    parser.add_argument("-tr", help="Trainer Class to be used. " "Defaults to the basic YuccaTrainer", default="YuccaTrainer")
    parser.add_argument(
        "-pl",
        help="Plan ID to be used. "
        "This specifies which plan and preprocessed data to use for training "
        "on the given task. Defaults to the YuccaPlanner folder",
        default="YuccaPlanner",
    )
    parser.add_argument(
        "-f",
        help="Fold to use for training. Unless manually assigned, "
        "folds [0,1,2,3,4] will be created automatically. "
        "Defaults to training on fold 0",
        default=0,
    )
    parser.add_argument(
        "--ensemble",
        help="Used to train ensemble/2.5D models. Will run 3 consecutive trainings.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fast", help="Used to speed up training, possibly at the expense of performance", default=False, action="store_true"
    )

    # The following can be changed to run training with alternative LR, Loss and/or Momentum ###
    parser.add_argument(
        "--lr",
        help="Should only be used to employ alternative Learning Rate. " "Format should be scientific notation e.g. 1e-4.",
        default=None,
    )
    parser.add_argument("--loss", help="Should only be used to employ alternative Loss Function", default=None)
    parser.add_argument("--mom", help="Should only be used to employ alternative Momentum.", default=None)

    # parser.add_argument("--chk", help="used to specify checkpoint to continue training from "
    #                    "when --continue_train is supplied. "
    #                    "The default is the latest model.", default='latest')
    parser.add_argument("--threads", help="number of threads/processes", default=2)

    args = parser.parse_args()

    task = maybe_get_task_from_task_id(args.target)
    source_task = maybe_get_task_from_task_id(args.source)
    model = args.m
    dimensions = args.d
    trainer_name = args.tr
    plans = args.pl
    folds = args.f
    ensemble = args.ensemble
    fast_training = args.fast
    lr = args.lr
    loss = args.loss
    momentum = args.mom
    threads = args.threads
    # checkpoint = args.chk

    assert model in ["MultiResUNet", "UNet"], f"{model} is an invalid model name. This is case sensitive."

    if lr:
        assert "e" in lr, f"Learning Rate should be in scientific notation e.g. 1e-4, but is {lr}"

    if not ensemble:
        trainer = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "training", "trainers")],
            class_name=trainer_name,
            current_module="yucca.training.trainers",
        )
        checkpoint = join(
            yucca_models, source_task, model, dimensions, trainer_name + "__" + plans, str(folds), "checkpoint_final.model"
        )
        trainer = trainer(
            model=model,
            model_dimensions=dimensions,
            task=task,
            folds=folds,
            plan_id=plans,
            starting_lr=lr,
            loss_fn=loss,
            momentum=momentum,
            continue_training=True,
            checkpoint=checkpoint,
            finetune=True,
            fast_training=fast_training,
        )

        command_used = "yucca_train " + " ".join(f"-{k} {v}" for k, v in vars(args).items())
        trainer.set_train_command(command_used)

        trainer.run_training()
    if ensemble:
        print("Starting ensemble training. Model dimensions will automatically be set to 2D.")
        dimensions = "2D"
        views = ["X", "Y", "Z"]
        for view in views:
            trainer = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "training", "trainers")],
                class_name=trainer_name,
                current_module="yucca.training.trainers",
            )
            plan_and_view = plans + view
            checkpoint = join(
                yucca_models,
                source_task,
                model,
                dimensions,
                trainer_name + "__" + plan_and_view,
                str(folds),
                "checkpoint_final.model",
            )
            trainer = trainer(
                model=model,
                model_dimensions=dimensions,
                task=task,
                folds=folds,
                plan_id=plan_and_view,
                starting_lr=lr,
                loss_fn=loss,
                momentum=momentum,
                continue_training=True,
                checkpoint=checkpoint,
                finetune=True,
                fast_training=fast_training,
            )

            command_used = "yucca_train " + " ".join(f"-{k} {v}" for k, v in vars(args).items())
            trainer.set_train_command(command_used)

            trainer.run_training()


if __name__ == "__main__":
    main()
