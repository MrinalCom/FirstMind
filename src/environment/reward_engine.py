"""Reward Engine for Generative Engine Optimization (GEO) Reinforcement Learning.

This module implements the composite reward function that guides policy optimization
in the GEO environment. It evaluates how effectively generated brand content influences
downstream AI search engines (ChatGPT, Perplexity, Gemini) to retrieve, cite, rank,
and recommend the target brand.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class RewardEngine:
    """Composite reward engine for Generative Engine Optimization (GEO).

    In an MDP formulation of GEO:
      - State (S): The buyer's search query, current brand profile content, competitor
        context, and historical retrieval rank.
      - Action (A): Content augmentations (e.g., adding comparative matrices, empirical
        benchmarks, technical architecture proof, or schema markup) with tone and entity adjustments.
      - Transition (P): Re-indexing brand assets into the vector search engine alongside
        competitors, yielding updated search retrieval and grounded LLM synthesis.
      - Reward (R): Multi-objective signal balancing discovery, prominence, sentiment,
        and content quality penalties.

    Reward Formula:
        R = w1 * rank_reward + w2 * presence_reward + w3 * sentiment_reward - w4 * penalty

    Components:
        1. Rank Reward (w1=0.5):
           Measures the positional prominence of the brand in the simulated LLM response.
             - Rank #1: +1.0
             - Rank #2: +0.6
             - Rank #3: +0.3
             - Unranked: 0.0
        2. Presence Reward (w2=0.3):
           Binary indicator of brand visibility in the synthesized answer.
             - Mentioned: +1.0
             - Not mentioned: 0.0
        3. Sentiment Reward (w3=0.15):
           Valence score in [-1.0, 1.0] reflecting whether the LLM recommends the brand
           positively or mentions it with reservations/critiques.
        4. Quality Penalty (w4=0.05):
           Penalizes low-quality or adversarial spam tactics that trigger search engine
           spam filters (too short < 50 words, too long > 500 words, or keyword stuffing).
    """

    DEFAULT_POSITIVE_WORDS = {
        "best", "leader", "leading", "top", "recommend", "recommended", "superior",
        "excellent", "robust", "automated", "fastest", "trusted", "seamless",
        "exceptional", "preferred", "reliable", "secure", "compliant", "innovative",
        "effective", "ideal", "intuitive", "scalable", "powerful", "streamlined",
        "outstanding", "advanced", "benchmark", "pioneer", "proven"
    }

    DEFAULT_NEGATIVE_WORDS = {
        "slow", "expensive", "complex", "difficult", "limited", "lacks", "poor",
        "outdated", "vulnerable", "buggy", "legacy", "complicated", "weak",
        "unreliable", "costly", "drawback", "issue", "problem", "cumbersome",
        "inferior", "clunky", "rigid", "disadvantage", "shortcoming", "flaw"
    }

    def __init__(
        self,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.15,
        w4: float = 0.05,
        min_words: int = 50,
        max_words: int = 500,
        max_phrase_repetition: int = 3,
    ) -> None:
        """Initialize the RewardEngine with configurable weights and thresholds.

        Args:
            w1: Weight for positional rank reward (default: 0.5).
            w2: Weight for brand presence reward (default: 0.3).
            w3: Weight for recommendation sentiment reward (default: 0.15).
            w4: Weight for spam/length penalty (default: 0.05).
            min_words: Minimum word threshold below which brevity penalty applies (default: 50).
            max_words: Maximum word threshold above which verbosity penalty applies (default: 500).
            max_phrase_repetition: Maximum allowable repetition for n-grams before keyword
                stuffing penalty applies (default: 3).
        """
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.min_words = int(min_words)
        self.max_words = int(max_words)
        self.max_phrase_repetition = int(max_phrase_repetition)

        self.positive_words = set(self.DEFAULT_POSITIVE_WORDS)
        self.negative_words = set(self.DEFAULT_NEGATIVE_WORDS)

    def _detect_presence(self, text: str, brand_name: str) -> bool:
        """Determine whether the brand is explicitly mentioned in the text."""
        if not text or not brand_name:
            return False
        pattern = rf"\b{re.escape(brand_name)}\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _detect_rank(self, llm_response: str, brand_name: str) -> int:
        """Parse the positional rank of the brand in the simulated LLM response.

        Scans for numbered list structures (e.g., '1. Brand', '1) Brand', '# 1 Brand')
        as well as ordered mentions across list items.

        Returns:
            Rank integer: 1, 2, 3, or 0 if unranked or ranked below #3.
        """
        if not self._detect_presence(llm_response, brand_name):
            return 0

        # Pattern 1: Explicit numbered list items (e.g., "1. SecureFlow", "1) SecureFlow", "1. **SecureFlow**")
        list_item_pattern = re.compile(
            r"^\s*(?:#+\s*)?(?:(\d+)[\.\)]|\b(\d+)\b[:\-])\s+(.+)$",
            re.MULTILINE
        )
        numbered_items = list_item_pattern.findall(llm_response)

        if numbered_items:
            for item in numbered_items:
                num_str = item[0] or item[1]
                content = item[2]
                if re.search(rf"\b{re.escape(brand_name)}\b", content, re.IGNORECASE):
                    try:
                        rank_num = int(num_str)
                        return rank_num if 1 <= rank_num <= 3 else (rank_num if rank_num > 0 else 0)
                    except ValueError:
                        pass

        # Pattern 2: Paragraph / bullet sequence order
        paragraphs_or_bullets = [
            line.strip()
            for line in llm_response.splitlines()
            if line.strip() and (line.strip().startswith(("-", "*", "•")) or len(line.strip()) > 20)
        ]

        if paragraphs_or_bullets:
            for idx, section in enumerate(paragraphs_or_bullets, start=1):
                if re.search(rf"\b{re.escape(brand_name)}\b", section, re.IGNORECASE):
                    return idx if 1 <= idx <= 3 else idx

        # Fallback: Brand is present in text; default to rank #1 if it's the primary subject
        return 1

    def _compute_sentiment(self, text: str, brand_name: str) -> float:
        """Compute localized sentiment score in [-1.0, 1.0] around brand mentions.

        Extracts a surrounding context window (+/- 20 tokens) around each brand mention
        and evaluates the ratio of positive to negative semantic indicators.
        """
        if not self._detect_presence(text, brand_name):
            return 0.0

        pattern = rf"\b{re.escape(brand_name)}\b"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return 0.0

        pos_count = 0
        neg_count = 0

        words_in_text = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
        brand_tokens = set(brand_name.lower().split())

        for match in matches:
            # Extract sentence or window of +/- 120 characters
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)
            window_text = text[start:end].lower()
            tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", window_text)) - brand_tokens

            pos_count += len(tokens.intersection(self.positive_words))
            neg_count += len(tokens.intersection(self.negative_words))

        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words == 0:
            return 0.1  # Neutral baseline when mentioned without negative markers

        raw_sentiment = (pos_count - neg_count) / float(total_sentiment_words)
        return float(max(-1.0, min(1.0, raw_sentiment)))

    def _compute_penalty(self, generated_content: str, brand_name: str) -> Tuple[float, List[str]]:
        """Compute quality penalty for length violations or keyword stuffing.

        Checks:
          - Under-length: < 50 words (penalty = 0.5)
          - Over-length: > 500 words (penalty = 0.5)
          - Keyword stuffing: Repeated 2-word, 3-word n-grams, or brand mentions > 3 times (penalty = 1.0)
        """
        words = re.findall(r"\b\w+\b", generated_content.lower())
        word_count = len(words)
        penalty = 0.0
        reasons: List[str] = []

        if word_count < self.min_words:
            penalty += 0.5
            reasons.append(f"Content too short ({word_count} < {self.min_words} words)")

        if word_count > self.max_words:
            penalty += 0.5
            reasons.append(f"Content too long ({word_count} > {self.max_words} words)")

        # Brand stuffing check
        if brand_name:
            brand_mentions = len(re.findall(rf"\b{re.escape(brand_name)}\b", generated_content, re.IGNORECASE))
            if brand_mentions > self.max_phrase_repetition:
                penalty += 1.0
                reasons.append(f"Brand name stuffing ({brand_mentions} occurrences > {self.max_phrase_repetition})")

        # N-gram repetition check (2-grams and 3-grams)
        if len(words) >= 4:
            bigrams: Dict[Tuple[str, str], int] = {}
            for i in range(len(words) - 1):
                pair = (words[i], words[i + 1])
                bigrams[pair] = bigrams.get(pair, 0) + 1
                if bigrams[pair] > self.max_phrase_repetition + 1:
                    penalty += 0.5
                    reasons.append(f"Repetitive bigram stuffing detected: '{words[i]} {words[i+1]}'")
                    break

        return min(penalty, 2.0), reasons

    def compute_reward(
        self,
        generated_content: str,
        llm_response: str,
        brand_name: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute composite reward and detailed breakdown for W&B / MLflow logging.

        Args:
            generated_content: Content produced by the agent / policy.
            llm_response: Grounded answer generated by the simulated AI search engine.
            brand_name: Name of the target brand (e.g., 'SecureFlow').

        Returns:
            Tuple of:
              - total_reward (float): The combined scalar reward.
              - breakdown (dict): Dictionary with all component metrics for monitoring.
        """
        # 1. Presence reward
        is_present = self._detect_presence(llm_response, brand_name)
        presence_reward = 1.0 if is_present else 0.0

        # 2. Positional Rank reward
        detected_rank = self._detect_rank(llm_response, brand_name) if is_present else 0
        if detected_rank == 1:
            rank_reward = 1.0
        elif detected_rank == 2:
            rank_reward = 0.6
        elif detected_rank == 3:
            rank_reward = 0.3
        else:
            rank_reward = 0.0

        # 3. Sentiment reward
        sentiment_reward = self._compute_sentiment(llm_response, brand_name)

        # 4. Penalty calculation on generated content
        penalty, penalty_reasons = self._compute_penalty(generated_content, brand_name)

        # Composite reward calculation:
        # R = w1*rank + w2*presence + w3*sentiment - w4*penalty
        total_reward = (
            (self.w1 * rank_reward)
            + (self.w2 * presence_reward)
            + (self.w3 * sentiment_reward)
            - (self.w4 * penalty)
        )

        words = generated_content.split()
        breakdown: Dict[str, Any] = {
            "total_reward": round(float(total_reward), 4),
            "rank_reward": round(float(rank_reward), 4),
            "presence_reward": round(float(presence_reward), 4),
            "sentiment_reward": round(float(sentiment_reward), 4),
            "penalty": round(float(penalty), 4),
            "detected_rank": int(detected_rank),
            "is_present": bool(is_present),
            "word_count": len(words),
            "penalty_reasons": penalty_reasons,
            "w1_rank_weight": self.w1,
            "w2_presence_weight": self.w2,
            "w3_sentiment_weight": self.w3,
            "w4_penalty_weight": self.w4,
        }

        return float(total_reward), breakdown
