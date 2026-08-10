from src.constants import PROJECT_PREFIX


def get_plot_blob_name(issue_time: str, trigger_status_bool: bool) -> str:
    return (
        f"{PROJECT_PREFIX}/monitoring/{issue_time}_{trigger_status_bool}.png"
    )


def get_email_subject(
    trigger_status: str, test: bool, monitoring_date: str
) -> str:
    test_text = "TEST : " if test else ""
    return (
        f"{test_text} Action antipatoire Tchad : inondations fluviales"
        f" - {trigger_status} {monitoring_date}"
    )
