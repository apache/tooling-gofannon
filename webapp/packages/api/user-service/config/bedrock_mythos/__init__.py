models = {
    # =========================================================================
    # Anthropic Claude Mythos Preview (via Bedrock Mantle).
    # 1M token context, 128K max output. Same adaptive thinking shape as
    # Opus 4.7+/4.8 (thinking={"type":"adaptive"}, output_config={"effort":"high"}).
    # llm_service.py's existing claude-opus substring detection in
    # _is_opus_4_7_or_later() does NOT match this model ID — if Mythos
    # needs the same adaptive-thinking code path, the detection function
    # may need updating. Worth verifying when Anthropic's setup email
    # confirms which thinking format Mythos expects.
    # =========================================================================
    "anthropic.claude-mythos-preview": {
        "returns_thoughts": True,
        "supports_effort": True,
        "supports_thinking": True,
        "context_window": 1000000,
        "parameters": {
            "temperature": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "description": "Randomness (0=focused, 1=creative). Locked to 1.0 when thinking enabled.",
                "mutually_exclusive_with": ["top_p"],
            },
            "top_p": {
                "type": "float",
                "default": 0.9,
                "min": 0.0,
                "max": 1.0,
                "description": "Nucleus sampling (0.1=conservative, 0.95=diverse)",
                "mutually_exclusive_with": ["temperature"],
            },
            "reasoning_effort": {
                "type": "choice",
                "default": "disable",
                "choices": ["disable", "low", "medium", "high", "xhigh", "max"],
                "description": "Reasoning Effort: Enables extended thinking. xhigh/max are Opus 4.7+ only.",
            },
            "max_tokens": {
                "type": "integer",
                "default": 16384,
                "min": 1,
                "max": 128000,
                "description": "Maximum tokens in response",
            },
        },
    },
}
