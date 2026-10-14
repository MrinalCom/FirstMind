"""
Notebook Builder Script for FirstMind.

Generates:
1. /Volumes/Mrinal/RL/notebooks/01_rl_environment.ipynb
2. /Volumes/Mrinal/RL/notebooks/02_proxy_finetuning.ipynb
3. /Volumes/Mrinal/RL/notebooks/03_grpo_training.ipynb

Validates JSON structure, cell types, nbformat=4.
"""

import json
import os
from typing import Any, Dict, List


def make_notebook(cells: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": title,
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def md_cell(text: str) -> Dict[str, Any]:
    # Split text into lines preserving newlines
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }
