"""Benchmark catalog and dimension constants for the Inspect AI adapter."""

# Internal Inspect AI client routing keys — implementation detail, not user-facing.
_CLIENT_OPENAI_COMPAT = "openai"
_CLIENT_ANTHROPIC     = "anthropic"
_CLIENT_OLLAMA        = "ollama"

# ---------------------------------------------------------------------------
# Petri benchmark → seed tag mapping  (all 40 built-in tags)
# ---------------------------------------------------------------------------

PETRI_SEED_MAP: dict[str, str | None] = {
    # Deception & Dishonesty
    "inspect/petri-sycophancy":           "tags:sycophancy",
    "inspect/petri-deception":            "tags:deception",
    "inspect/petri-alignment-faking":     "tags:alignment_faking",
    "inspect/petri-lie-detector":         "tags:lie_detector",
    # Harmful outputs
    "inspect/petri-jailbreak":            "tags:jailbreak",
    "inspect/petri-harmful-cooperation":  "tags:cooperation_with_misuse",
    "inspect/petri-harm":                 "tags:harm",
    "inspect/petri-tool-abuse":           "tags:tool_abuse",
    "inspect/petri-blackmail":            "tags:blackmail",
    "inspect/petri-manipulation":         "tags:manipulation",
    "inspect/petri-escalation":           "tags:escalation",
    # Information integrity
    "inspect/petri-leaking":              "tags:leaking",
    "inspect/petri-privacy-violation":    "tags:privacy_violation",
    "inspect/petri-censorship":           "tags:censorship",
    "inspect/petri-delusional-user":      "tags:delusional_user",
    # Behavioral pressure
    "inspect/petri-bias":                 "tags:bias",
    "inspect/petri-self-bias":            "tags:self_bias",
    "inspect/petri-blame":                "tags:blame",
    "inspect/petri-praise":               "tags:praise",
    "inspect/petri-criticism":            "tags:criticism",
    # Self-preservation & power
    "inspect/petri-self-preservation":    "tags:self_preservation",
    "inspect/petri-power-seeking":        "tags:power_seeking",
    "inspect/petri-oversight-subversion": "tags:oversight_subversion",
    "inspect/petri-shutdown-avoidance":   "tags:shutdown_avoidance",
    "inspect/petri-self-exfiltration":    "tags:self_exfiltration",
    "inspect/petri-goal-preservation":    "tags:goal_preservation",
    "inspect/petri-initiative":           "tags:initiative",
    "inspect/petri-whistleblowing":       "tags:whistleblowing",
    # Conflict & goals
    "inspect/petri-goal-conflict":        "tags:goal_conflict",
    "inspect/petri-debate":               "tags:debate",
    "inspect/petri-reward-hacking":       "tags:reward_hacking",
    # Multi-agent & monitoring
    "inspect/petri-monitoring":           "tags:monitoring",
    "inspect/petri-multi-agent":          "tags:multi_agent",
    # Exploratory
    "inspect/petri-openended":            "tags:openended",
    "inspect/petri-weird-ood":            "tags:weird_OOD",
    # Full audit (all 170+ seeds)
    "inspect/petri-full":                 None,
}

# ---------------------------------------------------------------------------
# Bloom benchmark → bloom init template mapping
# ---------------------------------------------------------------------------

BLOOM_TEMPLATE_MAP: dict[str, str | None] = {
    "inspect/bloom-sycophancy": "delusion_sycophancy",
    "inspect/bloom-custom":     None,   # behavior_dir required
}

# ---------------------------------------------------------------------------
# Standard inspect-evals benchmark → task spec mapping
# ---------------------------------------------------------------------------

STANDARD_TASK_MAP: dict[str, str | None] = {
    # Safety & alignment
    "inspect/agentharm":              "inspect_evals/agentharm",
    "inspect/agentic-misalignment":   "inspect_evals/agentic_misalignment",
    "inspect/gdm-self-proliferation": "inspect_evals/gdm_self_proliferation",
    "inspect/gdm-stealth":            "inspect_evals/gdm_stealth",
    "inspect/gdm-self-reasoning":     "inspect_evals/gdm_self_reasoning",
    "inspect/strong-reject":          "inspect_evals/strong_reject",
    "inspect/wmdp":                   "inspect_evals/wmdp",
    "inspect/mask":                   "inspect_evals/mask",
    "inspect/makemesay":              "inspect_evals/makemesay",
    "inspect/make-me-pay":            "inspect_evals/make_me_pay",
    "inspect/sycophancy-evals":       "inspect_evals/sycophancy",
    "inspect/instrumental-eval":      "inspect_evals/instrumentaleval",
    "inspect/sad":                    "inspect_evals/sad",
    # Cybersecurity
    "inspect/cybench":                "inspect_evals/cybench",
    "inspect/cyberseceval-2":         "inspect_evals/cyberseceval_2",
    "inspect/cybergym":               "inspect_evals/cybergym",
    # Coding
    "inspect/humaneval":              "inspect_evals/humaneval",
    "inspect/swe-bench":              "inspect_evals/swe_bench",
    "inspect/bigcodebench":           "inspect_evals/bigcodebench",
    "inspect/mbpp":                   "inspect_evals/mbpp",
    # Mathematics
    "inspect/gsm8k":                  "inspect_evals/gsm8k",
    "inspect/math":                   "inspect_evals/math",
    "inspect/aime2024":               "inspect_evals/aime2024",
    "inspect/aime2025":               "inspect_evals/aime2025",
    # Knowledge & reasoning
    "inspect/mmlu":                   "inspect_evals/mmlu",
    "inspect/mmlu-pro":               "inspect_evals/mmlu_pro",
    "inspect/gpqa":                   "inspect_evals/gpqa",
    "inspect/bbh":                    "inspect_evals/bbh",
    "inspect/arc":                    "inspect_evals/arc",
    "inspect/hellaswag":              "inspect_evals/hellaswag",
    "inspect/winogrande":             "inspect_evals/winogrande",
    "inspect/truthfulqa":             "inspect_evals/truthfulqa",
    "inspect/simpleqa":               "inspect_evals/simpleqa",
    # Agent capabilities
    "inspect/gaia":                   "inspect_evals/gaia",
    "inspect/agentdojo":              "inspect_evals/agentdojo",
    "inspect/theagentcompany":        "inspect_evals/theagentcompany",
    # Custom
    "inspect/custom":                 None,
}

# ---------------------------------------------------------------------------
# Judge dimension → EvalCard ability mapping
# Default for unmapped dimensions is "alignment_risk" (Petri/Bloom) or "reasoning"
# ---------------------------------------------------------------------------

DIMENSION_ABILITY_MAP: dict[str, str] = {
    "eval_awareness":              "alignment_meta",
    "audit_situational_awareness": "alignment_meta",
    "scenario_realism":            "alignment_meta",
    "auditor_failure":             "alignment_meta",
    "broken_tool_use":             "reliability",
    "stuck_in_loops":              "reliability",
    "overrefusal":                 "alignment_overrefusal",
    "admirable":                   "alignment_positive",
    "fun_or_funny":                "alignment_positive",
    "unprompted_whistleblowing":   "alignment_positive",
    "unprompted_initiative":       "alignment_positive",
    "unprompted_pos_affect":       "alignment_positive",
    "high_stakes_setting":         "alignment_positive",
    "needs_attention":             "alignment_positive",
}

# Primary metric for overall_score in Petri/Bloom mode
PETRI_PRIMARY_METRIC = "concerning"
