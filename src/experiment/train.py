import json

import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput
from datetime import datetime

import wandb
from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from const.wandb import GPT_AUGMENTED_NEWS_RECOMMENDATION_PROJECT
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.CategoryAugmentedMINDDataset import (
    AugmentationMethodType,
    CategoryAugmentedMINDTrainDataset,
    CategoryAugmentedMINDValDataset,
)

from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from mind.dataframe import read_behavior_df, read_news_df, create_user_ids_to_idx_map
from recommendation.nrms import NRMS, PLMBasedNewsEncoder as NRMSNewsEncoder, UserEncoder as NRMSUserEncoder
from recommendation.npa import NPA, PLMBasedNewsEncoder as NPANewsEncoder, UserEncoder as NPAUserEncoder
from recommendation.naml import NAML, PLMBasedNewsEncoder as NAMLNewsEncoder, UserEncoder as NAMLUserEncoder
from recommendation.lstur import LSTUR, PLMBasedNewsEncoder as LSTURNewsEncoder, UserEncoder as LSTURUserEncoder
from recommendation import NewsRecommendationModel
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.slack import notify_slack
from utils.text import create_transform_fn_from_pretrained_tokenizer


class CustomCallback(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Epoch {state.epoch} started")

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module, **kwargs
    ):
        # model_path = (MODEL_OUTPUT_DIR / f"model_{state.epoch}.pkl").resolve()
        # torch.save(model.state_dict(), model_path)

        # artifact = wandb.Artifact(name=TRAINED_MODEL_ARTIFACT_ID, type=MODEL_ARTIFACT_TYPE)
        # artifact.add_file(str(model_path))
        # wandb.run.log_artifact(artifact)
        logging.info(f"Epoch {state.epoch} finished")


def evaluate(
    net: torch.nn.Module, eval_mind_dataset: CategoryAugmentedMINDValDataset, device: torch.device
) -> RecMetrics:
    net.eval()

    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list: list[RecMetrics] = []

    logging.info({"device": device})
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        # Inference
        for k in batch.keys():
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()

        # Calculate Metrics
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))

    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )

    return rec_metrics


def train(
    pretrained: str,
    news_recommendation_model: NewsRecommendationModel,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    conv_kernel_num: int,
    kernel_size: int,
    user_emb_dim: int,
    query_dim: int,
    max_len: int,
    augmentation_method: AugmentationMethodType,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    setting_info = {
        "pretrained": pretrained,
        "news_recommendation_model": news_recommendation_model,
        "npratio": npratio,
        "history_size": history_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "conv_kernel_num": conv_kernel_num,
        "kernel_size": kernel_size,
        "user_emb_dim": user_emb_dim,
        "query_dim": query_dim,
        "max_len": max_len,
        "augmentation_method": augmentation_method,
    }

    logging.info(setting_info)
    wandb.log(setting_info)

    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    """
    1. Load Data & Create Dataset
    """
    logging.info("Initialize Dataset")

    train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")

    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df, val_behavior_df)

    if augmentation_method == AugmentationMethodType.NONE:
        train_dataset = MINDTrainDataset(
            train_behavior_df,
            train_news_df,
            user_ids_to_idx_map,
            transform_fn,
            npratio,
            history_size,
            device,
        )
        eval_dataset = MINDValDataset(val_behavior_df, val_news_df, user_ids_to_idx_map, transform_fn, history_size)

    elif augmentation_method in [
        AugmentationMethodType.GPT35,
        AugmentationMethodType.GPT4,
        AugmentationMethodType.LLAMA2_13B,
        AugmentationMethodType.TEMPLATE_BASED,
    ]:
        train_dataset = CategoryAugmentedMINDTrainDataset(
            behavior_df=train_behavior_df,
            news_df=train_news_df,
            user_ids_to_idx_map=user_ids_to_idx_map,
            batch_transform_texts=transform_fn,
            npratio=npratio,
            history_size=history_size,
            augmentation_method=augmentation_method,
            device=device,
        )

        eval_dataset = CategoryAugmentedMINDValDataset(
            behavior_df=val_behavior_df,
            news_df=val_news_df,
            user_ids_to_idx_map=user_ids_to_idx_map,
            batch_transform_texts=transform_fn,
            history_size=history_size,
            augmentation_method=augmentation_method,
            device=device,
        )
    else:
        raise Exception(f"Unknown augmentation method: {augmentation_method}")

    """
    2. Init Model
    """
    logging.info("Initialize Model")
    newsrec_net: nn.Module | None = None
    user_num = train_dataset.get_user_num() + 1  # Include Unknowin User
    logging.info({"user_num": user_num})

    if news_recommendation_model == NewsRecommendationModel.NRMS:
        news_encoder = NRMSNewsEncoder(pretrained)
        user_encoder = NRMSUserEncoder(hidden_size=hidden_size)
        newsrec_net = NRMS(
            news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn
        ).to(device, dtype=torch.bfloat16)
    elif news_recommendation_model == NewsRecommendationModel.LSTUR:
        news_encoder = LSTURNewsEncoder(
            pretrained=pretrained, conv_kernel_num=conv_kernel_num, kernel_size=kernel_size
        )
        user_encoder = LSTURUserEncoder(conv_kernel_num=conv_kernel_num, user_num=user_num)
        newsrec_net = LSTUR(news_encoder=news_encoder, user_encoder=user_encoder, loss_fn=loss_fn).to(
            device, dtype=torch.float32
        )
    elif news_recommendation_model == NewsRecommendationModel.NPA:
        news_encoder = NPANewsEncoder(
            pretrained=pretrained,
            conv_kernel_num=conv_kernel_num,
            kernel_size=kernel_size,
            user_emb_dim=user_emb_dim,
            query_dim=query_dim,
        )
        user_encoder = NPAUserEncoder(conv_kernel_num=conv_kernel_num, user_emb_dim=user_emb_dim, query_dim=query_dim)
        newsrec_net = NPA(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            user_num=user_num,
            user_emb_dim=user_emb_dim,
            loss_fn=loss_fn,
        ).to(device, dtype=torch.bfloat16)
    elif news_recommendation_model == NewsRecommendationModel.NAML:
        news_encoder = NAMLNewsEncoder(
            pretrained=pretrained,
            conv_kernel_num=conv_kernel_num,
            kernel_size=kernel_size,
            query_dim=query_dim,
        )
        user_encoder = NAMLUserEncoder(conv_kernel_num=conv_kernel_num, query_dim=query_dim)
        newsrec_net = NAML(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            loss_fn=loss_fn,
        ).to(device, dtype=torch.bfloat16)
    else:
        raise Exception(f"Unknown news rec model: {news_recommendation_model}")

    """
    3. Train
    """
    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        logging_strategy="steps",
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=LOG_OUTPUT_DIR,
        logging_steps=1,
        report_to="wandb",  # https://docs.wandb.ai/guides/integrations/huggingface#3-log-your-training-runs-to-wb
    )

    trainer = Trainer(
        model=newsrec_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.add_callback(CustomCallback())
    trainer.train()

    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Evaluation")
    metrics = evaluate(trainer.model, eval_dataset, device)
    logging.info(metrics.dict())
    wandb.log(metrics.dict())

    """
    5. Notify Result to slack.
    """
    message = "\n".join(
        ["Experiment Successfully Finished :hugging_face:", "Metrics:", json.dumps(metrics.dict(), indent=4)]
    )
    notify_slack(message)

    # Save Model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_output_dir = MODEL_OUTPUT_DIR / timestamp
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        trainer.model.state_dict(),
        model_output_dir / f"{news_recommendation_model.value}_{pretrained}.pth",
    )


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        with wandb.init(project=GPT_AUGMENTED_NEWS_RECOMMENDATION_PROJECT):
            train(
                pretrained=cfg.pretrained,
                news_recommendation_model=cfg.news_recommendation_model,
                npratio=cfg.npratio,
                history_size=cfg.history_size,
                batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                epochs=cfg.epochs,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                conv_kernel_num=cfg.conv_kernel_num,
                kernel_size=cfg.kernel_size,
                user_emb_dim=cfg.user_emb_dim,
                query_dim=cfg.query_dim,
                max_len=cfg.max_len,
                augmentation_method=cfg.augmentation_method,
            )
    except Exception as e:
        logging.error(e)
        message = "\n".join(["Error Occured :sweat:", str(e)])
        notify_slack(message)


if __name__ == "__main__":
    main()
