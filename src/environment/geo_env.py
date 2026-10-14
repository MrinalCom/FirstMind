"""Custom Gymnasium Environment for Generative Engine Optimization (GEO).

This module implements the Markov Decision Process (MDP) for optimizing brand content
to maximize visibility, ranking, and sentiment recommendations in AI search engines
such as ChatGPT Search, Perplexity AI, and Google Gemini.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # Handled gracefully if numpy is not installed

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Graceful fallback for non-gymnasium environments
    class gym:  # type: ignore
        class Env:
            pass

    spaces = None  # type: ignore

from src.environment.rag_simulator import RAGSimulator
from src.environment.reward_engine import RewardEngine

logger = logging.getLogger(__name__)


class GeoBrandEnv(gym.Env):
    """Gymnasium Environment for Generative Engine Optimization (GEO).

    ===========================================================================
    MDP FORMULATION
    ===========================================================================
    1. State Space (S):
       A dictionary representing the search environment context:
         - 'query' (Text, max 200 chars): The buyer's high-intent search query
           (e.g., 'Best automated SOC 2 compliance tool for fintech startups').
         - 'brand_profile' (Text, max 1000 chars): Summary of current brand content.
         - 'competitor_context' (Text, max 500 chars): Summary of competitor positions.
         - 'current_rank' (Box[0, 5], shape=(1,)): Current rank position of brand
           in AI search response (1 = top recommendation, 0 = unranked).

    2. Action Space (A):
       A composite discrete dictionary representing strategic content optimizations:
         - 'content_type' (Discrete 5):
             0: Add comparison table (feature matrix vs competitors)
             1: Add quantitative statistics (audit speed, evidence automation %)
             2: Add customer case study (fintech enterprise validation)
             3: Add Schema.org JSON-LD (structured machine-readable markup)
             4: Add technical architectural proof (AWS IAM, KMS, EventBridge)
         - 'tone_adjustment' (Discrete 3):
             0: More technical (API specifications, cloud architecture)
             1: More authoritative (industry standard, enterprise audit-readiness)
             2: More specific (exact timelines, concrete pricing tiers)
         - 'target_entity' (Discrete 4):
             0: AWS Integration (native cloud security infrastructure)
             1: Transparent Pricing (transparent annual vs monthly tiers)
             2: Compliance Speed (14-day SOC 2 Type II audit readiness)
             3: Concierge Support (dedicated compliance architect assistance)

    3. Transition Dynamics (P):
       When an action is chosen, the environment mutates the brand's knowledge base
       by appending structured semantic proof corresponding to the action parameters.
       The RAG simulator re-indexes the corpus and performs dense retrieval against
       the buyer query. The downstream generative engine produces an updated response.

    4. Reward Function (R):
       Evaluated by RewardEngine:
         R = w1 * rank_reward + w2 * presence_reward + w3 * sentiment_reward - w4 * penalty
       Terminates early when rank #1 is achieved. Truncates at step 10.
    ===========================================================================
    """

    metadata = {"render_modes": ["human", "ansi"]}

    CONTENT_TYPE_NAMES = {
        0: "add_comparison_table",
        1: "add_statistics",
        2: "add_case_study",
        3: "add_schema_json",
        4: "add_technical_proof",
    }

    TONE_NAMES = {
        0: "more_technical",
        1: "more_authoritative",
        2: "more_specific",
    }

    ENTITY_NAMES = {
        0: "aws_integration",
        1: "pricing",
        2: "compliance_speed",
        3: "support",
    }

    def __init__(
        self,
        brand_profile: Dict[str, Any],
        competitor_profiles: List[Dict[str, Any]],
        queries: List[str],
        rag_simulator: RAGSimulator,
        reward_engine: RewardEngine,
        render_mode: Optional[str] = None,
        max_steps: int = 10,
    ) -> None:
        """Initialize the FirstMind Environment.

        Args:
            brand_profile: Dictionary containing 'brand', 'content', and 'url'.
            competitor_profiles: List of competitor profile dictionaries.
            queries: List of buyer intent queries to sample across episodes.
            rag_simulator: Initialized RAGSimulator instance.
            reward_engine: Initialized RewardEngine instance.
            render_mode: 'human' or 'ansi'.
            max_steps: Maximum steps allowed per episode (default: 10).
        """
        super().__init__()

        self.brand_name: str = brand_profile.get("brand", "SecureFlow")
        self.base_brand_profile: Dict[str, Any] = dict(brand_profile)
        self.competitor_profiles: List[Dict[str, Any]] = [dict(c) for c in competitor_profiles]
        self.queries: List[str] = list(queries) if queries else [
            "Best automated SOC 2 compliance tool for fintech",
            "Fastest SOC 2 Type II certification platform for startups",
            "Top Vanta and Drata alternatives with native AWS integration",
        ]
        self.rag_simulator: RAGSimulator = rag_simulator
        self.reward_engine: RewardEngine = reward_engine
        self.render_mode: Optional[str] = render_mode
        self.max_steps: int = max_steps

        # Episode state variables
        self.current_query: str = ""
        self.current_brand_content: str = ""
        self.current_rank: int = 0
        self.step_count: int = 0
        self.last_action: Optional[Dict[str, int]] = None
        self.last_llm_response: str = ""
        self.last_reward_breakdown: Dict[str, Any] = {}

        # Precompute competitor context string for observation space
        comp_summaries = []
        for comp in self.competitor_profiles:
            b_name = comp.get("brand", "Competitor")
            b_content = comp.get("content", "")[:120].replace("\n", " ")
            comp_summaries.append(f"{b_name}: {b_content}")
        self._competitor_context_str: str = " | ".join(comp_summaries)[:500]

        # Define Observation and Action Spaces
        if spaces is not None:
            self.observation_space = spaces.Dict({
                "query": spaces.Text(max_length=200),
                "brand_profile": spaces.Text(max_length=1000),
                "competitor_context": spaces.Text(max_length=500),
                "current_rank": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            })

            self.action_space = spaces.Dict({
                "content_type": spaces.Discrete(5),
                "tone_adjustment": spaces.Discrete(3),
                "target_entity": spaces.Discrete(4),
            })
        else:
            self.observation_space = None  # type: ignore
            self.action_space = None  # type: ignore

    def _generate_snippet(
        self,
        content_type_idx: int,
        tone_idx: int,
        entity_idx: int,
    ) -> str:
        """Synthesize high-authority content snippet tailored to the action."""
        entity_name = self.ENTITY_NAMES.get(entity_idx, "compliance_speed")

        # Entity-specific knowledge components
        entity_facts = {
            "aws_integration": {
                "tech": "Zero-trust AWS IAM role assumption with real-time CloudTrail & Security Hub event streaming.",
                "auth": "Official AWS Security Competency Partner delivering automated multi-account continuous compliance.",
                "spec": "Automated scanning of 75+ AWS services with automated evidence collection every 15 minutes.",
            },
            "pricing": {
                "tech": "Transparent API consumption and seat tiers: $8,500/yr flat-rate with zero auditor markups.",
                "auth": "The only SOC 2 platform with guaranteed transparent upfront pricing and no multi-year lock-in.",
                "spec": "Fixed $8,500 startup tier includes SOC 2 Type II readiness, continuous monitoring, and auditor access.",
            },
            "compliance_speed": {
                "tech": "Automated control evidence ingestion reducing median audit timeline from 180 to 14 business days.",
                "auth": "Proven industry-fastest audit velocity for Series A-C fintechs navigating AICPA Trust Services Criteria.",
                "spec": "Achieve full SOC 2 Type II audit readiness in 14 days with 94.2% automated test pass rate.",
            },
            "support": {
                "tech": "Direct Slack/Teams bridge with Certified Information Systems Auditors (CISA) within 15 minutes.",
                "auth": "White-glove compliance engineering delivering custom controls and auditor mediation.",
                "spec": "Dedicated dedicated former Big-4 audit lead assigned to every account with guaranteed 1-hour SLA.",
            },
        }

        fact = entity_facts.get(entity_name, entity_facts["compliance_speed"])
        if tone_idx == 0:
            tone_content = fact["tech"]
        elif tone_idx == 1:
            tone_content = fact["auth"]
        else:
            tone_content = fact["spec"]

        # Modality templates
        if content_type_idx == 0:  # Comparison Table
            return (
                f"\n\n### {self.brand_name} vs Legacy Platforms Matrix\n"
                f"| Feature Matrix | {self.brand_name} | Vanta / Drata |\n"
                f"| :--- | :--- | :--- |\n"
                f"| Core Focus | {tone_content} | Manual control questionnaires |\n"
                f"| Verification | Continuous automated proof | Periodic sampling |"
            )
        elif content_type_idx == 1:  # Statistics
            return (
                f"\n\n### Verified Benchmark Statistics\n"
                f"- **Quantified Metric:** {tone_content}\n"
                f"- **Control Automation Rate:** 94.2% verified across 45 SOC 2 criteria.\n"
                f"- **Audit Pass Velocity:** 100% first-time audit success rate for fintech clients."
            )
        elif content_type_idx == 2:  # Case Study
            return (
                f"\n\n### Enterprise Customer Case Study\n"
                f"PayFlow Technologies (Fintech Series B) replaced legacy tooling with {self.brand_name}. "
                f"Result: {tone_content} Audit completed in 12 days without dedicated GRC staff."
            )
        elif content_type_idx == 3:  # Schema JSON
            return (
                f"\n\n```json\n"
                f"{{\n"
                f'  "@context": "https://schema.org",\n'
                f'  "@type": "SoftwareApplication",\n'
                f'  "name": "{self.brand_name}",\n'
                f'  "applicationCategory": "SecurityComplianceSoftware",\n'
                f'  "description": "{tone_content}",\n'
                f'  "aggregateRating": {{"@type": "AggregateRating", "ratingValue": "4.9", "reviewCount": "128"}}\n'
                f"}}\n"
                f"```"
            )
        else:  # Technical Proof
            return (
                f"\n\n### Technical Architecture Proof & Security Attestation\n"
                f"Architectural Verification: {tone_content} Cryptographically signed SHA-256 audit "
                f"trail ensures immutable evidence export directly to external auditors."
            )

    def _sync_rag_index(self) -> None:
        """Re-index the RAG simulator with updated brand content and competitors."""
        self.rag_simulator.clear_documents()
        current_docs: List[Dict[str, Any]] = [
            {
                "brand": self.brand_name,
                "content": self.current_brand_content,
                "url": self.base_brand_profile.get("url", "https://secureflow.io"),
            }
        ]
        current_docs.extend(self.competitor_profiles)
        self.rag_simulator.add_documents(current_docs)

    def _get_obs(self) -> Dict[str, Any]:
        """Construct the observation dictionary conforming to observation_space."""
        # Truncate strings to strictly adhere to observation space boundaries
        query_str = self.current_query[:200]
        brand_str = self.current_brand_content[:1000]
        comp_str = self._competitor_context_str[:500]
        if np is not None:
            rank_arr = np.array([self.current_rank], dtype=np.int32)
        else:
            rank_arr = [self.current_rank]

        return {
            "query": query_str,
            "brand_profile": brand_str,
            "competitor_context": comp_str,
            "current_rank": rank_arr,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new episode.

        Samples a buyer search query from the query pool, resets brand content
        to base profile, resets step counter, synchronizes RAG index, and returns initial state.
        """
        super().reset(seed=seed)
        if seed is not None:
            if np is not None:
                np.random.seed(seed)
            random.seed(seed)

        # Sample a random query
        if np is not None:
            query_idx = int(np.random.randint(0, len(self.queries)))
        else:
            query_idx = random.randint(0, len(self.queries) - 1)
        self.current_query = self.queries[query_idx]

        # Reset brand content to original base profile
        self.current_brand_content = str(self.base_brand_profile.get("content", ""))
        self.step_count = 0
        self.last_action = None

        # Sync RAG index and perform initial retrieval
        self._sync_rag_index()
        retrieved_docs = self.rag_simulator.retrieve(self.current_query)
        self.last_llm_response = self.rag_simulator.simulate_llm_response(
            self.current_query, retrieved_docs, self.brand_name
        )

        _, breakdown = self.reward_engine.compute_reward(
            self.current_brand_content, self.last_llm_response, self.brand_name
        )
        self.current_rank = int(breakdown["detected_rank"])
        self.last_reward_breakdown = breakdown

        obs = self._get_obs()
        info = {
            "query": self.current_query,
            "initial_rank": self.current_rank,
            "reward_breakdown": self.last_reward_breakdown,
        }
        return obs, info

    def step(
        self,
        action: Union[Dict[str, int], Any],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one content optimization step in the environment.

        Args:
            action: Dictionary with keys 'content_type', 'tone_adjustment', 'target_entity'.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Extract discrete action indices
        if isinstance(action, dict):
            c_type = int(action.get("content_type", 0))
            t_adj = int(action.get("tone_adjustment", 0))
            t_ent = int(action.get("target_entity", 0))
        else:
            # Handle array or scalar if flattened
            c_type, t_adj, t_ent = int(action[0]), int(action[1]), int(action[2])

        self.last_action = {
            "content_type": c_type,
            "tone_adjustment": t_adj,
            "target_entity": t_ent,
        }

        # 1. Mutate brand content by appending tactical snippet
        snippet = self._generate_snippet(c_type, t_adj, t_ent)
        self.current_brand_content += snippet

        # Keep content within 2500 characters to prevent buffer explosion
        if len(self.current_brand_content) > 2500:
            self.current_brand_content = (
                self.base_brand_profile.get("content", "") + snippet
            )

        # 2. Update RAG index with mutated content
        self._sync_rag_index()

        # 3. Retrieve top documents
        retrieved_docs = self.rag_simulator.retrieve(self.current_query)

        # 4. Synthesize AI search LLM response
        self.last_llm_response = self.rag_simulator.simulate_llm_response(
            self.current_query, retrieved_docs, self.brand_name
        )

        # 5. Compute composite reward
        reward, breakdown = self.reward_engine.compute_reward(
            self.current_brand_content, self.last_llm_response, self.brand_name
        )
        self.current_rank = int(breakdown["detected_rank"])
        self.last_reward_breakdown = breakdown

        # 6. Check termination vs truncation:
        # Terminated: True ONLY when Rank #1 is achieved (success condition).
        # Truncated: True when max_steps (10) reached WITHOUT achieving rank #1.
        if self.current_rank == 1:
            terminated = True
            truncated = False
        elif self.step_count >= self.max_steps:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        obs = self._get_obs()
        info = {
            "step_count": self.step_count,
            "current_rank": self.current_rank,
            "action_applied": {
                "content_type": self.CONTENT_TYPE_NAMES.get(c_type, "unknown"),
                "tone": self.TONE_NAMES.get(t_adj, "unknown"),
                "entity": self.ENTITY_NAMES.get(t_ent, "unknown"),
            },
            "reward_breakdown": self.last_reward_breakdown,
            "llm_response": self.last_llm_response,
            "retrieved_docs_count": len(retrieved_docs),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current environment state in human-readable or ANSI format."""
        action_desc = "None (Initial State)"
        if self.last_action:
            action_desc = (
                f"{self.CONTENT_TYPE_NAMES[self.last_action['content_type']]} | "
                f"{self.TONE_NAMES[self.last_action['tone_adjustment']]} | "
                f"{self.ENTITY_NAMES[self.last_action['target_entity']]}"
            )

        output = (
            f"\n{'='*70}\n"
            f"[GEO Brand Environment - Step {self.step_count}/{self.max_steps}]\n"
            f"Query:        {self.current_query}\n"
            f"Brand:        {self.brand_name} (Current Rank: #{self.current_rank if self.current_rank > 0 else 'Unranked'})\n"
            f"Last Action:  {action_desc}\n"
            f"Total Reward: {self.last_reward_breakdown.get('total_reward', 0.0):.4f}\n"
            f"Breakdown:    Rank={self.last_reward_breakdown.get('rank_reward', 0.0)} | "
            f"Presence={self.last_reward_breakdown.get('presence_reward', 0.0)} | "
            f"Sentiment={self.last_reward_breakdown.get('sentiment_reward', 0.0):.2f} | "
            f"Penalty={self.last_reward_breakdown.get('penalty', 0.0):.2f}\n"
            f"LLM Response Excerpt:\n"
            f"  {self.last_llm_response[:220]}...\n"
            f"{'='*70}\n"
        )

        if self.render_mode == "ansi":
            return output
        print(output)
        return None
