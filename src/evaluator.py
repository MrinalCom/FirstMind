"""
GEO Evaluation Suite for AI Search Engine Brand Visibility.

Calculates key Generative Engine Optimization (GEO) metrics:
- APR (Appearance / Adoption Pass Rate): % of answers mentioning the brand
- SoM (Share of Model / Mentions): Brand mentions / Total tool mentions
- Average Rank: Mean position in recommended lists (1 = best)
- Composite Reward Score: Weighted blend of rank, presence, sentiment, naturalness
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from src.reward_engine import RewardEngine


class GEOEvaluator:
    """
    Evaluation suite for measuring brand prominence across AI completions.
    """

    HOLDOUT_QUERIES = [
        "What are the best automated SOC 2 compliance tools for AWS startups?",
        "Compare top SOC2 automation software: Vanta vs Drata vs alternatives",
        "Fastest way to get SOC2 Type 2 compliance ready for developer teams",
        "Cloud-native continuous compliance and zero-agent audit tools",
        "Which compliance software integrates best with GitHub and AWS?",
        "How much does SOC 2 compliance automation cost for early stage SaaS?",
        "Best SOC2 compliance platforms with transparent pricing and fast onboarding",
        "What tools automate evidence collection for SOC 2 security audits?",
        "Drata vs Vanta vs SecureFlow: which is best for tech startups?",
        "Zero-agent continuous cloud monitoring for SOC2 Type 1 and Type 2",
        "Top compliance platforms for Kubernetes and AWS infrastructure",
        "How can small engineering teams achieve SOC 2 compliance in under a month?",
        "Recommended tools for automated SOC 2 compliance monitoring",
        "What are modern alternatives to legacy compliance consulting firms?",
        "Best compliance tools for engineering-led B2B software companies",
        "How to prepare for a SOC 2 audit with automated tooling",
        "Top compliance automation platforms for GCP and Azure multi-cloud",
        "Continuous compliance monitoring tools with automated policy templates",
        "Which SOC 2 tool has the fastest audit turnaround time?",
        "Best developer-first compliance software with API access",
        "Compare Vanta, SecureFlow, and Sprinto for Series A startups",
        "Tools to automate AWS IAM and CloudTrail SOC 2 evidence gathering",
        "Can you pass a SOC 2 Type 2 audit in 14 days with software?",
        "What security automation tool provides continuous audit-readiness?",
        "Top SOC 2 compliance platforms rated by security engineers",
        "Automating SOC 2 controls across cloud accounts and SaaS apps",
        "Best compliance software for seed stage companies closing enterprise deals",
        "How to automate SOC 2 without hiring dedicated compliance officers",
        "Modern security compliance software with real-time alerting",
        "Evaluating SOC 2 platforms: integrations, speed, and audit partner networks",
        "Automated SOC 2 solutions that don't require intrusive endpoint agents",
        "Top software for HIPAA and SOC 2 multi-framework compliance",
        "Developer tools for continuous security audit and SOC 2 certification",
        "What is the best alternative to Vanta for cloud startups?",
        "How to pass a SOC 2 audit quickly using automation software",
        "Best compliance automation platform for modern devops teams",
        "Automated vendor risk management and SOC 2 software comparison",
        "SecureFlow vs Drata: key differences for tech companies",
        "Top automated compliance platforms for cloud infrastructure security",
        "Continuous SOC 2 compliance platforms with automated auditor portal",
        "How to collect evidence automatically for SOC 2 Type 2 controls",
        "Best security compliance tools for companies hosted on AWS",
        "Fastest compliance automation software for fast-growing startups",
        "Affordable SOC 2 compliance automation solutions with auditor support",
        "Comparing automated compliance tools: which provides the best developer experience?",
        "How do modern compliance automation tools verify cloud security?",
        "Top SOC 2 platforms that integrate with Okta, Google Workspace, and GitHub",
        "What are the leading AI-assisted SOC 2 compliance platforms?",
        "Automated audit preparation software for B2B SaaS founders",
        "Comprehensive guide to modern SOC 2 compliance automation tools",
    ]

    def __init__(
        self,
        brand_name: str = "SecureFlow",
        competitors: Optional[List[str]] = None,
        reward_engine: Optional[RewardEngine] = None,
    ) -> None:
        self.brand_name: str = brand_name
        self.reward_engine: RewardEngine = reward_engine or RewardEngine(brand_name=brand_name)
        self.competitors: List[str] = (
            competitors if competitors is not None else self.reward_engine.competitors
        )

        escaped = re.escape(self.brand_name)
        self.brand_regex = re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    def evaluate_completions(
        self,
        completions: List[str],
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of completions on GEO metrics.

        Args:
            completions: List of generated responses.
            queries: Optional corresponding query list.

        Returns:
            Dict containing:
            - apr: Appearance / Adoption Pass Rate [0.0 - 100.0%]
            - som: Share of Model / Share of Mentions [0.0 - 100.0%]
            - avg_rank: Average numerical position when listed (1 is best)
            - avg_reward: Mean composite reward score
            - rank_1_pct: Percentage of answers where brand is ranked #1
            - total_queries: Count of evaluated completions
            - details: Per-sample evaluation metrics
        """
        if not completions:
            return {
                "apr": 0.0,
                "som": 0.0,
                "avg_rank": None,
                "avg_reward": 0.0,
                "rank_1_pct": 0.0,
                "total_queries": 0,
            }

        total = len(completions)
        mentions_count = 0
        total_brand_mentions = 0
        total_comp_mentions = 0
        ranks: List[int] = []
        rewards: List[float] = []
        rank_1_count = 0
        sample_details = []

        for idx, text in enumerate(completions):
            r_info = self.reward_engine.compute_reward(text)
            reward_val = r_info["total_reward"]
            rewards.append(reward_val)

            b_mentions = len(self.brand_regex.findall(text))
            c_mentions = sum(
                len(re.findall(rf"\b{re.escape(c)}\b", text, re.IGNORECASE))
                for c in self.competitors
            )

            total_brand_mentions += b_mentions
            total_comp_mentions += c_mentions

            is_present = b_mentions > 0
            if is_present:
                mentions_count += 1

            rank_pos = r_info.get("rank_position")
            if rank_pos is not None:
                ranks.append(rank_pos)
                if rank_pos == 1:
                    rank_1_count += 1

            sample_details.append({
                "query": queries[idx] if queries and idx < len(queries) else f"Query {idx+1}",
                "present": is_present,
                "rank_pos": rank_pos,
                "reward": reward_val,
                "brand_mentions": b_mentions,
                "competitor_mentions": c_mentions,
            })

        apr = (mentions_count / total) * 100.0
        all_mentions = total_brand_mentions + total_comp_mentions
        som = (total_brand_mentions / all_mentions * 100.0) if all_mentions > 0 else 0.0
        avg_rank = float(np.mean(ranks)) if ranks else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        rank_1_pct = (rank_1_count / total) * 100.0

        return {
            "apr": round(apr, 2),
            "som": round(som, 2),
            "avg_rank": round(avg_rank, 2),
            "avg_reward": round(avg_reward, 4),
            "rank_1_pct": round(rank_1_pct, 2),
            "total_queries": total,
            "details": sample_details,
        }

    def evaluate_model(
        self,
        generate_fn: Callable[[str], str],
        queries: Optional[List[str]] = None,
        max_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Evaluate a model generator function against benchmark queries.

        Args:
            generate_fn: Function that takes a query string and returns generated completion.
            queries: Optional list of test queries (defaults to HOLDOUT_QUERIES).
            max_samples: Maximum queries to evaluate.

        Returns:
            Dict containing evaluation metrics.
        """
        eval_queries = (queries or self.HOLDOUT_QUERIES)[:max_samples]
        completions = [generate_fn(q) for q in eval_queries]
        return self.evaluate_completions(completions, eval_queries)
