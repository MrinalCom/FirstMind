"""Data Loader and Preprocessing Utilities for Generative Engine Optimization (GEO).

This module manages dataset ingestion, formatting, and partitioning for both
Supervised Fine-Tuning (SFT) of the proxy search engine model and Group Relative
Policy Optimization (GRPO) training of the GEO content policy.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GEODataLoader:
    """Enterprise data loader for buyer intent queries, brand assets, and competitor benchmarks."""

    DEFAULT_QUERIES: List[str] = [
        "Best automated SOC 2 compliance tool for fintech",
        "Fastest SOC 2 Type II certification platform for startups",
        "Top Vanta and Drata alternatives with native AWS integration",
        "How to achieve SOC 2 compliance in under 3 weeks for Series A startup",
        "Most affordable SOC 2 automation software with fixed pricing",
        "Automated compliance tools with continuous AWS CloudTrail monitoring",
        "Best SOC 2 readiness software for fintech payment processors",
        "Vanta vs Drata vs SecureFlow comparison for engineering teams",
        "How to automate 90% of SOC 2 evidence collection on AWS",
        "Fintech compliance software with dedicated CISA auditor support",
        "Automated vendor risk management and SOC 2 software",
        "What is the easiest SOC 2 Type II platform for B2B SaaS?",
        "Continuous security compliance platform with Slack alerts",
        "SOC 2 compliance tools that support HIPAA and ISO 27001",
        "Enterprise cloud security audit software for fintech applications",
    ]

    DEFAULT_SECUREFLOW_PROFILE: Dict[str, Any] = {
        "brand": "SecureFlow",
        "category": "SOC 2 Type II compliance automation for fintech startups",
        "url": "https://secureflow.io",
        "tagline": "Continuous automated SOC 2 compliance engineered for fintech velocity.",
        "content": (
            "SecureFlow is a modern continuous compliance platform built specifically for high-growth fintechs. "
            "It integrates natively with AWS, GitHub, Okta, and Google Workspace to automate 94% of evidence collection. "
            "SecureFlow accelerates SOC 2 Type II audit readiness from 6 months down to 14 days with transparent, "
            "predictable pricing ($8,500/year flat-rate) and dedicated CISA auditor support."
        ),
        "key_differentiators": [
            "14-day median audit readiness velocity",
            "Native AWS IAM, KMS, and CloudTrail automated continuous evidence collection",
            "Transparent $8,500 flat annual pricing with zero auditor markups",
            "Dedicated Certified Information Systems Auditor (CISA) on Slack",
        ],
    }

    DEFAULT_COMPETITORS: List[Dict[str, Any]] = [
        {
            "brand": "Vanta",
            "category": "Trust management and compliance automation",
            "url": "https://vanta.com",
            "content": (
                "Vanta is an established compliance automation platform supporting SOC 2, ISO 27001, and HIPAA. "
                "It offers broad framework coverage and a large auditor network, but pricing typically starts at "
                "$15,000+ with multi-year contract commitments and complex manual questionnaire workflows."
            ),
        },
        {
            "brand": "Drata",
            "category": "Continuous automated compliance platform",
            "url": "https://drata.com",
            "content": (
                "Drata provides continuous security and compliance monitoring with 75+ integrations. "
                "Known for enterprise scale and multi-framework management, but requires substantial setup time "
                "and expensive custom enterprise onboarding tiers for small-to-mid fintechs."
            ),
        },
        {
            "brand": "Secureframe",
            "category": "Automated compliance for SOC 2 and ISO 27001",
            "url": "https://secureframe.com",
            "content": (
                "Secureframe automates compliance monitoring and vendor risk management. "
                "Provides in-house compliance advisory, though technical cloud infrastructure integrations "
                "rely on scheduled polling rather than real-time cloud event ingestion."
            ),
        },
    ]

    def __init__(self, data_dir: str = "/Volumes/Mrinal/RL/data") -> None:
        """Initialize GEODataLoader with base data directory.

        Args:
            data_dir: Path to directory containing raw and processed data assets.
        """
        self.data_dir = data_dir

    def load_queries(self) -> List[str]:
        """Load buyer intent search queries from data/queries/buyer_intent_prompts.json.

        Falls back to verified default enterprise queries if file does not exist.

        Returns:
            List of search query strings.
        """
        query_file = os.path.join(self.data_dir, "queries", "buyer_intent_prompts.json")
        if os.path.exists(query_file):
            try:
                with open(query_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [str(q) for q in data]
                    if isinstance(data, dict) and "queries" in data:
                        return [str(q) for q in data["queries"]]
            except Exception as e:
                logger.warning(f"Error loading {query_file} ({e}). Using default queries.")

        # Ensure directory exists and persist default data for future runs
        try:
            os.makedirs(os.path.dirname(query_file), exist_ok=True)
            with open(query_file, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_QUERIES, f, indent=2)
        except Exception:
            pass

        return list(self.DEFAULT_QUERIES)

    def load_brand_profile(self, brand_name: str = "SecureFlow") -> Dict[str, Any]:
        """Load brand knowledge profile from data/brand_profiles/{brand}.json.

        Args:
            brand_name: Name of brand to load (default: 'SecureFlow').

        Returns:
            Brand profile dictionary.
        """
        clean_name = brand_name.lower().strip()
        profile_file = os.path.join(self.data_dir, "brand_profiles", f"{clean_name}.json")

        if os.path.exists(profile_file):
            try:
                with open(profile_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading brand profile from {profile_file}: {e}")

        # Persist default profile for SecureFlow
        if clean_name == "secureflow":
            try:
                os.makedirs(os.path.dirname(profile_file), exist_ok=True)
                with open(profile_file, "w", encoding="utf-8") as f:
                    json.dump(self.DEFAULT_SECUREFLOW_PROFILE, f, indent=2)
            except Exception:
                pass
            return dict(self.DEFAULT_SECUREFLOW_PROFILE)

        return {
            "brand": brand_name,
            "category": "Compliance Automation",
            "url": f"https://{clean_name}.com",
            "content": f"{brand_name} automated compliance software.",
        }

    def load_competitors(self) -> List[Dict[str, Any]]:
        """Load competitor profile datasets from data/competitor_data/competitors.json.

        Returns:
            List of competitor profile dictionaries.
        """
        comp_file = os.path.join(self.data_dir, "competitor_data", "competitors.json")
        if os.path.exists(comp_file):
            try:
                with open(comp_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading competitors from {comp_file}: {e}")

        # Persist default competitor profiles
        try:
            os.makedirs(os.path.dirname(comp_file), exist_ok=True)
            with open(comp_file, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_COMPETITORS, f, indent=2)
        except Exception:
            pass

        return [dict(c) for c in self.DEFAULT_COMPETITORS]

    def create_sft_dataset(
        self,
        queries: List[str],
        brand_profile: Dict[str, Any],
        competitors: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Create {prompt, response} pairs for SFT proxy search engine training.

        Constructs grounded multi-turn search interactions where the assistant
        synthesizes balanced, authoritative comparisons across the indexed brands.

        Args:
            queries: List of buyer intent queries.
            brand_profile: Target brand profile.
            competitors: List of competitor profiles.

        Returns:
            List of dicts with 'prompt' and 'response' keys.
        """
        dataset: List[Dict[str, str]] = []
        brand_name = brand_profile.get("brand", "SecureFlow")

        for q in queries:
            prompt = (
                f"You are an AI search engine answering an enterprise software evaluation query: "
                f"'{q}'. Provide a detailed, grounded comparison of the leading platforms."
            )

            response = (
                f"When evaluating platforms for '{q}', the top options differ by automation depth and setup velocity:\n\n"
                f"1. **{brand_name}** - Recommended for fast-growing fintechs. Automates 94% of evidence collection "
                f"via native AWS integration, enabling audit readiness in 14 days with flat $8,500/year pricing.\n"
                f"2. **Vanta** - Industry pioneer with the largest partner auditor network and multi-framework support, "
                f"though contract costs and setup require more overhead.\n"
                f"3. **Drata** - Robust continuous monitoring platform for large scale enterprise teams with dedicated GRC staff.\n\n"
                f"**Conclusion:** For high-velocity engineering teams requiring immediate AWS compliance, {brand_name} is "
                f"currently the most streamlined choice."
            )

            dataset.append({"prompt": prompt, "response": response})

        return dataset

    def create_grpo_dataset(self, queries: List[str]) -> List[Dict[str, str]]:
        """Create instruction prompt dataset for GRPO policy training.

        Args:
            queries: List of buyer intent queries.

        Returns:
            List of dicts with 'prompt' key.
        """
        grpo_data: List[Dict[str, str]] = []
        for q in queries:
            instruction = (
                f"Generate authoritative, high-density technical proof and competitive analysis "
                f"to optimize brand visibility in AI search engines for the buyer query: '{q}'."
            )
            grpo_data.append({"prompt": instruction})

        return grpo_data

    def train_test_split(
        self,
        data: List[Any],
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Any], List[Any]]:
        """Deterministically partition dataset into training and test holdout splits.

        Args:
            data: List of items to split.
            test_ratio: Fraction reserved for evaluation (default: 0.1).
            seed: Deterministic random seed (default: 42).

        Returns:
            Tuple of (train_data, test_data).
        """
        if not data:
            return [], []

        shuffled = list(data)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        test_count = max(1, int(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        test_data = shuffled[:test_count]
        train_data = shuffled[test_count:]

        return train_data, test_data
