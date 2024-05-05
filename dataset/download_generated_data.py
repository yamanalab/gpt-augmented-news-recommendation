# Download Generated Text from Remote Artifact
import wandb
from src.const.path import MIND_GENERATED_DATASET_DIR
from src.const.wandb import GENERATED_ARTIFACT_ID, GPT_AUGMENTED_NEWS_RECOMMENDATION_PROJECT


if __name__ == "__main__":
    with wandb.init(project=GPT_AUGMENTED_NEWS_RECOMMENDATION_PROJECT) as run:
        # Download data from artifact.
        artifact = run.use_artifact(
            f"{GENERATED_ARTIFACT_ID}:latest",
            type="dataset",
        )
        MIND_GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        artifact.download(root=MIND_GENERATED_DATASET_DIR)
