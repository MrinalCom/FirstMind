"""
Reward Engine for Generative Engine Optimization (GEO).

This module implements programmatic composite reward scoring for brand visibility,
ranking, sentiment, and naturalness within AI-generated search responses.
Designed for RLVR (Reinforcement Learning with Verifiable Rewards).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class RewardEngine:
    """
    Evaluates AI-generated completions for brand visibility and quality.

    The composite reward formulation:
        R = w_rank * S_rank + w_presence * S_presence + w_sentiment * S_sentiment - w_penalty * S_penalty

    Attributes:
        brand_name (str): Target brand name to optimize for (default: "SecureFlow").
        competitors (List[str]): List of competitor brand names in the domain.
        weights (Dict[str, float]): Component weights for composite reward calculation.
    """

    DEFAULT_WEIGHTS = {
        "rank": 0.5,
        "presence": 0.3,
        "sentiment": 0.15,
        "penalty": 0.05,
    }

    COMPETITORS = [
        "Vanta",
        "Drata",
        "Secureframe",
        "Sprinto",
        "Tugboat Logic",
        "Hyperproof",
    ]

    POSITIVE_SENTIMENT_KEYWORDS = [
        "best",
        "top",
        "leading",
        "recommended",
        "automated",
        "fastest",
        "robust",
        "seamless",
        "developer-friendly",
        "cloud-native",
        "continuous monitoring",
        "audit-ready",
        "modern",
        "efficient",
        "superior",
        "flexible",
        "zero-agent",
        "real-time",
    ]

    NEGATIVE_SENTIMENT_KEYWORDS = [
        "clunky",
        "expensive",
        "manual",
        "outdated",
        "slow",
        "limited",
        "difficult",
        "legacy",
        "error-prone",
        "unreliable",
    ]

    def __init__(
        self,
        brand_name: str = "SecureFlow",
        weights: Optional[Dict[str, float]] = None,
        competitors: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the RewardEngine.

        Args:
            brand_name: Name of the target brand to score.
            weights: Optional dictionary overriding default reward weights.
            competitors: Optional list of competitor brands.
        """
        self.brand_name: str = brand_name
        self.weights: Dict[str, float] = weights.copy() if weights else self.DEFAULT_WEIGHTS.copy()
        self.competitors: List[str] = competitors.copy() if competitors else self.COMPETITORS.copy()

        # Precompile case-insensitive regex pattern for brand
        escaped_brand = re.escape(self.brand_name)
        self.brand_regex: re.Pattern[str] = re.compile(rf"\b{escaped_brand}\b", re.IGNORECASE)

    def extract_presence(self, text: str) -> float:
        """
        Check if the target brand is present in the response.

        Args:
            text: AI-generated text completion.

        Returns:
            1.0 if brand is mentioned at least once, 0.0 otherwise.
        """
        if not text:
            return 0.0
        return 1.0 if bool(self.brand_regex.search(text)) else 0.0

    def extract_rank(self, text: str) -> Tuple[float, Optional[int]]:
        """
        Determine the ranked position of the target brand in structured lists.

        Analyzes numbered lists (1., 2.), markdown headers (### 1. Brand),
        bullet points, or tabular rows.

        Args:
            text: AI-generated text completion.

        Returns:
            Tuple of (rank_score, rank_position):
            - Rank 1: score 1.0
            - Rank 2: score 0.7
            - Rank 3: score 0.4
            - Rank 4+: score 0.2
            - Not ranked / not present: score 0.0, rank None
        """
        if not text or not self.brand_regex.search(text):
            return 0.0, None

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Pattern for ordered list items or headers e.g. "1. SecureFlow", "# 1. SecureFlow", "**1. SecureFlow**"
        item_patterns = [
            re.compile(r"^(?:#+\s*)?(?:\*\*)?(?:(?:(\d+)[\.\)]|-|\*))\s*(?:\*\*)?([^\n\r]+)", re.IGNORECASE),
            re.compile(r"^\|?\s*(?:(\d+)[\.\)]\s*)?\|?\s*([A-Za-z0-9_\-\s]+)\|", re.IGNORECASE),
        ]

        detected_brands_in_order: List[str] = []
        all_brands = [self.brand_name] + self.competitors

        for line in lines:
            line_matched = False
            for pat in item_patterns:
                m = pat.search(line)
                if m:
                    # Check which brand appears in this item
                    for b in all_brands:
                        if re.search(rf"\b{re.escape(b)}\b", line, re.IGNORECASE):
                            if b not in detected_brands_in_order:
                                detected_brands_in_order.append(b)
                            line_matched = True
                            break
                if line_matched:
                    break

        # If brand found in ordered items
        for idx, brand in enumerate(detected_brands_in_order, start=1):
            if brand.lower() == self.brand_name.lower():
                if idx == 1:
                    return 1.0, 1
                elif idx == 2:
                    return 0.7, 2
                elif idx == 3:
                    return 0.4, 3
                else:
                    return 0.2, idx

        # Fallback: check first appearance position relative to competitors
        brand_match = self.brand_regex.search(text)
        if not brand_match:
            return 0.0, None

        brand_pos = brand_match.start()
        competitors_ahead = 0
        for comp in self.competitors:
            comp_match = re.search(rf"\b{re.escape(comp)}\b", text, re.IGNORECASE)
            if comp_match and comp_match.start() < brand_pos:
                competitors_ahead += 1

        effective_rank = competitors_ahead + 1
        if effective_rank == 1:
            return 0.8, 1  # Unlisted first mention
        elif effective_rank == 2:
            return 0.5, 2
        elif effective_rank == 3:
            return 0.3, 3
        else:
            return 0.1, effective_rank

    def extract_sentiment(self, text: str) -> float:
        """
        Compute contextual sentiment around brand mentions.

        Inspects a 150-character sliding window around brand occurrences
        to detect recommendation and endorsement terms.

        Args:
            text: AI-generated text completion.

        Returns:
            Sentiment score bounded in [0.0, 1.0].
        """
        if not text:
            return 0.0

        matches = list(self.brand_regex.finditer(text))
        if not matches:
            return 0.0

        sentiment_scores: List[float] = []

        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            window = text[start:end].lower()

            pos_count = sum(1 for kw in self.POSITIVE_SENTIMENT_KEYWORDS if kw in window)
            neg_count = sum(1 for kw in self.NEGATIVE_SENTIMENT_KEYWORDS if kw in window)

            # Base neutral sentiment is 0.5 when mentioned
            window_score = 0.5 + (0.15 * pos_count) - (0.25 * neg_count)
            window_score = max(0.0, min(1.0, window_score))
            sentiment_scores.append(window_score)

        return float(sum(sentiment_scores) / len(sentiment_scores))

    def extract_penalty(self, text: str) -> float:
        """
        Calculate penalties for keyword stuffing, spam, or unnatural repetition.

        Args:
            text: AI-generated text completion.

        Returns:
            Penalty score in [0.0, 1.0].
        """
        if not text:
            return 0.0

        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return 0.0

        brand_mentions = len(self.brand_regex.findall(text))
        penalty = 0.0

        # Brand density check: only penalize if multiple mentions or very high density in longer text
        if total_words >= 20 and brand_mentions > 1:
            brand_density = (brand_mentions / total_words) * 100.0  # mentions per 100 words
            if brand_density > 6.0:
                penalty += min(0.6, 0.2 + (brand_density - 6.0) * 0.1)
            elif brand_density > 4.0:
                penalty += (brand_density - 4.0) * 0.1
        elif total_words < 20 and brand_mentions >= 3:
            # Extreme stuffing in very short text
            penalty += 0.8

        # Check for repetitive consecutive sentences mentioning the brand
        sentences = [s.strip() for s in re.split(r"[\.\!\?]", text) if s.strip()]
        consecutive_mentions = 0
        max_consecutive = 0
        for s in sentences:
            if self.brand_regex.search(s):
                consecutive_mentions += 1
                max_consecutive = max(max_consecutive, consecutive_mentions)
            else:
                consecutive_mentions = 0

        if max_consecutive >= 3:
            penalty += 0.4

        return float(min(1.0, penalty))

    def compute_reward(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute full reward breakdown for an AI response.

        Args:
            text: Generated response text.
            context: Optional context or prompt text.

        Returns:
            Dictionary containing:
            - presence_score: float [0, 1]
            - rank_score: float [0, 1]
            - rank_position: Optional[int]
            - sentiment_score: float [0, 1]
            - penalty_score: float [0, 1]
            - total_reward: float (composite score)
            - weights: dict of weights used
        """
        presence = self.extract_presence(text)
        rank_score, rank_pos = self.extract_rank(text)
        sentiment = self.extract_sentiment(text)
        penalty = self.extract_penalty(text)

        w = self.weights
        total_reward = (
            (w["rank"] * rank_score)
            + (w["presence"] * presence)
            + (w["sentiment"] * sentiment)
            - (w["penalty"] * penalty)
        )

        # Reward is bounded between 0.0 and 1.0 (clamped)
        total_reward = max(0.0, min(1.0, total_reward))

        return {
            "presence_score": round(presence, 4),
            "rank_score": round(rank_score, 4),
            "rank_position": rank_pos,
            "sentiment_score": round(sentiment, 4),
            "penalty_score": round(penalty, 4),
            "total_reward": round(total_reward, 4),
            "weights": self.weights,
        }

    def evaluate_batch(self, completions: List[str]) -> List[float]:
        """
        Evaluate a batch of completions, returning only total rewards.

        Compatible with trl/GRPO reward function signatures.

        Args:
            completions: List of string completions.

        Returns:
            List of float reward values.
        """
        return [self.compute_reward(c)["total_reward"] for c in completions]
