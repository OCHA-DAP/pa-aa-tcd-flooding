import io
import os
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

import ocha_stratus as stratus
from dotenv import load_dotenv
from html2text import html2text
from jinja2 import Environment, FileSystemLoader

from src.monitoring import etl, utils

load_dotenv()

STATIC_DIR = Path("src/monitoring/email/static/")
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

    # Send emails if activated, if within warning threshold,
    # or if it is a Monday, or if testing
    if activations or warnings or monitoring_date_obj.weekday() == 0 or test:
        print(f"Sending emails for date: {monitoring_date}")
        # always send informational, but only send trigger when triggering
        for email_type in activations + ["informational"]:
            print(f"Sending {email_type} email")
            ocha_logo_cid = make_msgid(domain="humdata.org")
            chart_cid = make_msgid(domain="humdata.org")

            environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            template = environment.get_template(f"{email_type}.html")

            distribution_list_name = (
                "info" if email_type == "informational" else "trigger"
            )
            distribution = utils.process_distribution_list(
                test, distribution_list_name
            )

            msg = EmailMessage()
            msg.set_charset("utf-8")
            msg["Subject"] = utils.get_email_subject(
                trigger_status, test, monitoring_date
            )
            msg["From"] = Address(
                "Centre de données humanitaires OCHA",
                utils.EMAIL_ADDRESS.split("@")[0],
                utils.EMAIL_ADDRESS.split("@")[1],
            )
            msg["To"] = [
                Address(
                    row["name"],
                    row["email"].split("@")[0],
                    row["email"].split("@")[1],
                )
                for _, row in distribution["to"].iterrows()
            ]
            msg["Cc"] = [
                Address(
                    row["name"],
                    row["email"].split("@")[0],
                    row["email"].split("@")[1],
                )
                for _, row in distribution["cc"].iterrows()
            ]

            html_str = template.render(
                pub_date=monitoring_date,
                ocha_logo_cid=ocha_logo_cid[1:-1],
                chart_cid=chart_cid[1:-1],  # Don't need if triggering
                test_email=test,
                trigger_status=trigger_status,
            )

            text_str = html2text(html_str)
            msg.set_content(text_str)
            msg.add_alternative(html_str, subtype="html")

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
                msg.get_payload()[1].add_related(
                    image_data.read(), "image", "png", cid=chart_cid
                )

            for filename, cid in zip(
                ["ocha_logo_wide.png"],
                [ocha_logo_cid],
            ):
                img_path = STATIC_DIR / filename
                with open(img_path, "rb") as img:
                    msg.get_payload()[1].add_related(
                        img.read(), "image", "png", cid=cid
                    )

            utils.send_email(msg, distribution["to"], distribution["cc"])
    else:
        print(f"Not sending email. Trigger status is {trigger_status}")
