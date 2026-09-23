import base64
import io
import os
from datetime import datetime
from pathlib import Path

import ocha_stratus as stratus
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from ocha_relay.listmonk import ListmonkClient

from src.constants import (
    ALWAYS_EMAIL,
    EMAIL_BACKEND,
    LISTMONK_INFO_LIST_ID,
    LISTMONK_TEST_LIST_ID,
    LISTMONK_TRIGGER_LIST_ID,
    SES_RECIPIENTS_LIVE,
    SES_RECIPIENTS_TEST,
    STAGE,
    TEST_EMAIL,
)
from src.monitoring import etl, utils
from src.ses_mail import recipients_from_env, send_via_ses

load_dotenv()

TEMPLATES_DIR = Path("src/monitoring/email/templates/")

if __name__ == "__main__":
    test = TEST_EMAIL or STAGE != "prod"
    if test:
        print("This is a TEST email!")
    print(f"Email backend: {EMAIL_BACKEND} (ALWAYS_EMAIL={ALWAYS_EMAIL})")
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

    if (
        activations
        or warnings
        or monitoring_date_obj.weekday() == 0
        or test
        or ALWAYS_EMAIL
    ):
        print(f"Sending emails for date: {monitoring_date}")
        client = ListmonkClient.from_env() if EMAIL_BACKEND == "listmonk" else None
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
                blob_client = stratus.get_container_client(
                    "projects", STAGE
                ).get_blob_client(blob_name)
                blob_client.download_blob().readinto(image_data)
                image_data.seek(0)
                if client is not None:
                    chart_url = client.upload_media(
                        image_data.read(),
                        f"tcd-flooding-{monitoring_date}.png",
                    )
                else:
                    # SES path: embed as a data URI; send_via_ses turns it
                    # into a CID inline attachment.
                    b64 = base64.b64encode(image_data.read()).decode()
                    chart_url = f"data:image/png;base64,{b64}"

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

            if client is None:
                recipients = recipients_from_env(
                    SES_RECIPIENTS_TEST if test else SES_RECIPIENTS_LIVE,
                    "SES_TEST_RECIPIENTS" if test else "SES_RECIPIENTS",
                )
                send_via_ses(subject, html_str, recipients, text_fallback=subject)
                print(f"Sent {email_type} email via SES to {recipients}")
            else:
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
