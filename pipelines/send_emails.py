import io
import os
from datetime import datetime
from pathlib import Path

import ocha_stratus as stratus
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from ocha_relay.listmonk import ListmonkClient

from src.constants import (
    LISTMONK_INFO_LIST_ID,
    LISTMONK_TEST_LIST_ID,
    LISTMONK_TRIGGER_LIST_ID,
)
from src.monitoring import etl, utils

load_dotenv()

TEMPLATES_DIR = Path("src/monitoring/email/templates/")
STAGE = os.getenv("STAGE", "dev")

if __name__ == "__main__":
    test = False if STAGE == "prod" else True
    if test:
        print("This is a TEST email!")
    monitoring_date = os.getenv("MONITORING_DATE", "")
    if not monitoring_date:
        monitoring_date = datetime.today().strftime("%Y-%m-%d")

    monitoring_date_obj = datetime.strptime(monitoring_date, "%Y-%m-%d")

    activations = etl.check_results(monitoring_date, activation=True)
    warnings = etl.check_results(monitoring_date, activation=False)
    trigger_status = "NON ACTIVÉ"
    if "readiness" in activations:
        trigger_status = "MOBILISATION ACTIVÉ"
    if "action" in activations:
        trigger_status = "ACTION ACTIVÉ"

    if activations or warnings or monitoring_date_obj.weekday() == 0 or test:
        print(f"Sending emails for date: {monitoring_date}")
        client = ListmonkClient.from_env()
        environment = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

        for email_type in activations + ["informational"]:
            print(f"Sending {email_type} email")

            if test:
                list_id = LISTMONK_TEST_LIST_ID
            elif email_type == "informational":
                list_id = LISTMONK_INFO_LIST_ID
            else:
                list_id = LISTMONK_TRIGGER_LIST_ID

            chart_url = None
            if email_type == "informational":
                blob_name = utils.get_plot_blob_name(
                    monitoring_date, bool(activations)
                )
                image_data = io.BytesIO()
                blob_client = stratus.get_container_client().get_blob_client(
                    blob_name
                )
                blob_client.download_blob().readinto(image_data)
                image_data.seek(0)
                chart_url = client.upload_media(
                    image_data.read(),
                    f"tcd-flooding-{monitoring_date}.png",
                )

            trigger_status_for_email = (
                "MOBILISATION ACTIVÉ"
                if email_type == "readiness"
                else trigger_status
            )

            template = environment.get_template(f"{email_type}.html")
            html_str = template.render(
                pub_date=monitoring_date,
                chart_url=chart_url,
                trigger_status=trigger_status_for_email,
            )

            subject = utils.get_email_subject(
                trigger_status_for_email, test, monitoring_date
            )
            test_prefix = "[TEST] " if test else ""
            slug = f"[FR] tcd-flooding-{email_type}-{monitoring_date}"
            campaign_name = test_prefix + slug

            campaign_id = client.create_campaign(
                name=campaign_name,
                subject=subject,
                body=html_str,
                list_ids=[list_id],
            )
            client.send_campaign(campaign_id, skip_confirmation=True)
            print(f"Sent {email_type} campaign (id={campaign_id})")
    else:
        print(f"Not sending email. Trigger status is {trigger_status}")
