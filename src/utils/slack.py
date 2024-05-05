from slack_sdk.webhook import WebhookClient, WebhookResponse

from const.env import SLACK_WEBHOOK_URL


def notify_slack(text: str) -> WebhookResponse:
    webhook = WebhookClient(SLACK_WEBHOOK_URL)
    return webhook.send(text=text)
