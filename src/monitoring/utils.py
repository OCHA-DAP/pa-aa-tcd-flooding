import os
import re
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage
from email.policy import SMTP as SMTP_POLICY

import ocha_stratus as stratus
import pandas as pd
from dotenv import load_dotenv

from src.constants import PROJECT_PREFIX

load_dotenv()

EMAIL_HOST = os.getenv("DSCI_AWS_EMAIL_HOST")
EMAIL_PORT = int(os.getenv("DSCI_AWS_EMAIL_PORT", 465))
EMAIL_PASSWORD = os.getenv("DSCI_AWS_EMAIL_PASSWORD")
EMAIL_USERNAME = os.getenv("DSCI_AWS_EMAIL_USERNAME")
EMAIL_ADDRESS = os.getenv("DSCI_AWS_EMAIL_ADDRESS")

UNI_SPACES = "\u00A0\u2007\u202F"
BAD_TOKENS = ("=?utf-8?q?=2C?=", "=?utf-8?q?=3B?=")  # encoded ',' or ';'


def process_distribution_list(test_list, email_type):
    distribution_list = get_distribution_list(test_list)
    valid_distribution_list = distribution_list[
        distribution_list["email"].apply(is_valid_email)
    ]
    invalid_distribution_list = distribution_list[
        ~distribution_list["email"].apply(is_valid_email)
    ]
    if not invalid_distribution_list.empty:
        print(
            f"Invalid emails found in distribution list: "
            f"{invalid_distribution_list['email'].tolist()}"
        )
    to_list = valid_distribution_list[
        valid_distribution_list[email_type] == "to"
    ]
    cc_list = valid_distribution_list[
        valid_distribution_list[email_type] == "cc"
    ]
    return {"to": to_list, "cc": cc_list}


def get_distribution_list(test_list) -> pd.DataFrame:
    """Load distribution list from blob storage."""
    if test_list:
        print("Using test distribution list")
        blob_name = f"{PROJECT_PREFIX}/email/test_distribution_list.csv"
    else:
        blob_name = f"{PROJECT_PREFIX}/email/distribution_list.csv"
    return stratus.load_csv_from_blob(blob_name)


def is_valid_email(email):
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_regex, email))


def get_plot_blob_name(issue_time, trigger_status_bool):
    return (
        f"{PROJECT_PREFIX}/monitoring/{issue_time}_{trigger_status_bool}.png"
    )


def send_email(msg, to_list, cc_list):
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        # check for bad tokens in headers
        # (it is possible that these will only show up once the message
        # has been built with the full list of recipients)
        # note that this is LIKELY redundant with find_bad_transition,
        # but is still useful since the actual issue was so hard to identify
        hdrs = msg.as_string(policy=SMTP_POLICY).split("\n\n", 1)[0]
        # uncomment below to see the whole thing, if you need to look where
        # the problem characters are, but these should have already been
        # flagged by find_bad_transition()
        # print(hdrs)
        if "=?utf-8?q?=2C?=" in hdrs or "=?utf-8?q?=3B?=" in hdrs:
            print(
                "Encoded comma/semicolon detected in To/Cc headers. "
                "Likely a trailing ',' or ';' or NBSP in a display name."
                " Patching before sending..."
            )

            _BAD_COMMA = re.compile(
                rb"=\?utf-8\?q\?=2c\?=" rb"(?:\s*|\r?\n[ \t]+)?",
                flags=re.IGNORECASE,
            )

            def headers_only_bytes(m, policy=SMTP_POLICY):
                data = m.as_bytes(policy=policy)  # CRLF endings
                head, _, tail = data.partition(b"\r\n\r\n")
                return head, b"\r\n\r\n", tail

            head, sep, body = headers_only_bytes(msg, policy=SMTP_POLICY)

            # Replace any standalone encoded comma with a literal ", "
            patched_head = _BAD_COMMA.sub(b", ", head)

            patched_bytes = patched_head + sep + body

            # send *patched_bytes*; your envelope (recipients) stays unchanged
            server.sendmail(
                EMAIL_ADDRESS,
                to_list["email"].tolist() + cc_list["email"].tolist(),
                patched_bytes,
            )
        else:
            server.sendmail(
                EMAIL_ADDRESS,
                to_list["email"].tolist() + cc_list["email"].tolist(),
                msg.as_string(),
            )
    print("Email sent!")


def get_email_subject(trigger_status, test, monitoring_date):
    test_text = "TEST : " if test else ""
    return (
        f"{test_text} Action antipatoire Tchad : inondations fluviales"
        f" - {trigger_status} {monitoring_date}"
    )


def _build_to_cc_header(to_rows, cc_rows):
    m = EmailMessage(policy=SMTP_POLICY)
    m["From"] = "noreply@example.com"
    if to_rows:
        m["To"] = [
            Address(str(r["name"] or ""), *str(r["email"]).split("@", 1))
            for r in to_rows
        ]
    if cc_rows:
        m["Cc"] = [
            Address(str(r["name"] or ""), *str(r["email"]).split("@", 1))
            for r in cc_rows
        ]
    m["Subject"] = "x"
    return m.as_string().split("\n\n", 1)[0]  # headers only


def _contains_bad_token(headers: str) -> bool:
    return any(tok in headers for tok in BAD_TOKENS)


def find_bad_transition(df_to, df_cc):
    """Find the first row in To or Cc that causes bad token in header.
    Note that if a row is flagged, the easiest way to fix it is simply to
    add whitespace in the name field in the distribution list so the line no
    longer tries to break at the comma, which creates a non-breaking comma
    instead of a normal one.
    """
    to = df_to.reset_index(drop=True)
    cc = df_cc.reset_index(drop=True)

    # 1) Grow To only; find first k where it breaks
    first_bad_k = None
    for k in range(1, len(to) + 1):
        hdrs = _build_to_cc_header(to.iloc[:k].to_dict("records"), [])
        if _contains_bad_token(hdrs):
            first_bad_k = k
            break

    if first_bad_k is not None:
        i_bad = first_bad_k - 1
        prev = to.iloc[i_bad - 1] if i_bad - 1 >= 0 else None
        cur = to.iloc[i_bad]
        print("[TO] Break occurs when adding row", i_bad)
        if prev is not None:
            print(
                "  Prev:",
                {"i": i_bad - 1, "name": prev["name"], "email": prev["email"]},
            )
        print(
            "  Curr:", {"i": i_bad, "name": cur["name"], "email": cur["email"]}
        )
        return ("TO", i_bad - 1 if prev is not None else None, i_bad)

    # 2) If To alone is fine, freeze all To and grow Cc
    for k in range(1, len(cc) + 1):
        hdrs = _build_to_cc_header(
            to.to_dict("records"), cc.iloc[:k].to_dict("records")
        )
        if _contains_bad_token(hdrs):
            j_bad = k - 1
            prev = cc.iloc[j_bad - 1] if j_bad - 1 >= 0 else None
            cur = cc.iloc[j_bad]
            print("[CC] Break occurs when adding row", j_bad)
            if prev is not None:
                print(
                    "  Prev:",
                    {
                        "j": j_bad - 1,
                        "name": prev["name"],
                        "email": prev["email"],
                    },
                )
            print(
                "  Curr:",
                {"j": j_bad, "name": cur["name"], "email": cur["email"]},
            )
            return ("CC", j_bad - 1 if prev is not None else None, j_bad)

    print("No transition found (header OK).")
    return None
