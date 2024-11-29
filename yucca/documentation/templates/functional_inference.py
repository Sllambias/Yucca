if __name__ == "__main__":
    import lightning as L
    import os
    import torch
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p as ensure_dir_exists
    from yucca.paths import (
        get_models_path,
        get_results_path,
        get_preprocessed_data_path,
        get_raw_data_path,
    )
    from yucca.modules.callbacks.prediction_writer import WritePredictionFromLogits
    from yucca.modules.lightning_modules.BaseLightningModule import BaseLightningModule
    from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
    from yucca.modules.data.datasets.YuccaDataset import YuccaTestPreprocessedDataset
    from yucca.pipeline.evaluation.YuccaEvaluator import YuccaEvaluator
    from yucca.documentation.templates.template_config import config

    ckpt_path = os.path.join(
        get_models_path(),
        config["task"],
        config["model_name"] + "__" + config["model_dimensions"],
        "__" + config["config_name"],
        "default",
        "kfold_5_fold_0",
        "version_0",
        "checkpoints",
        "last.ckpt",
    )

    gt_path = os.path.join(get_raw_data_path(), config["task"], "labelsTs")
    target_data_path = os.path.join(get_preprocessed_data_path(), config["task"] + "_test", "demo")

    save_path = os.path.join(
        get_results_path(),
        config["task"],
        config["task"],
        config["model_name"] + "__" + config["model_dimensions"],
        "__" + config["config_name"],
        "kfold_5_fold_0",
        "version_0",
        "best",
    )
    ensure_dir_exists(save_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    pred_writer = WritePredictionFromLogits(output_dir=save_path, save_softmax=False, write_interval="batch")

    model_module = BaseLightningModule(
        model=config["model"],
        model_dimensions=config["model_dimensions"],
        num_classes=config["num_classes"],
        num_modalities=config["num_modalities"],
        patch_size=config["patch_size"],
        crop_to_nonzero=config["crop_to_nonzero"],
        disable_inference_preprocessing=True,
    )

    data_module = YuccaDataModule(
        batch_size=config["batch_size"],
        patch_size=config["patch_size"],
        pred_save_dir=save_path,
        pred_data_dir=target_data_path,
        overwrite_predictions=True,
        image_extension="pt",
        test_dataset_class=YuccaTestPreprocessedDataset,
    )

    trainer = L.Trainer(
        callbacks=pred_writer,
        precision="32",
        enable_progress_bar=True,
    )

    trainer.predict(
        model=model_module,
        dataloaders=data_module,
        ckpt_path=ckpt_path,
        return_predictions=False,
    )

    evaluator = YuccaEvaluator(
        labels=config["classes"],
        folder_with_ground_truth=gt_path,
        folder_with_predictions=save_path,
        use_wandb=False,
    )
    evaluator.run()
