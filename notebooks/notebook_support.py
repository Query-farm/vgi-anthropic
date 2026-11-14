"""Presentation and fixture setup; accounting analysis stays in notebook SQL."""

from __future__ import annotations

import os
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

from synthetic_admin_api import START, SyntheticAdminApi

COLORS = {"Production": "#3D7B68", "Staging": "#5C86AF", "Default workspace": "#CB843D"}
SOURCE = "Source: synthetic local Admin API · real vgi-anthropic worker via Haybarn / VGI"


def configure(report_days: int):
    if type(report_days) is not int or not 1 <= report_days <= 7:
        raise ValueError("REPORT_DAYS must be an integer from 1 to 7; the fixture begins 2026-09-01 UTC")
    fixture = SyntheticAdminApi().start()
    previous_url = os.environ.get("VGI_ANTHROPIC_BASE_URL")
    os.environ["VGI_ANTHROPIC_BASE_URL"] = fixture.url
    return fixture, previous_url, START.isoformat(), (START + timedelta(days=report_days)).isoformat()


def close_fixture(fixture, previous_url):
    fixture.stop()
    if previous_url is None:
        os.environ.pop("VGI_ANTHROPIC_BASE_URL", None)
    else:
        os.environ["VGI_ANTHROPIC_BASE_URL"] = previous_url


def prose(text: str) -> None:
    display(Markdown(text))


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        palette=list(COLORS.values()),
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str) -> None:
    fig.text(0.09, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.09, 0.91, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.10, 1, 0.83))
    fig.text(0.09, 0.03, SOURCE, fontsize=8, color="#6C7D73")
    plt.show()
