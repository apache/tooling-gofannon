# Mythos preview model configuration (accessed via AWS Bedrock Mantle).
#
# Mantle is a distinct Bedrock endpoint
# (https://bedrock-mantle.{region}.api.aws/v1) with OpenAI-compatible
# APIs and its own IAM action namespace (bedrock-mantle:*). Mythos is
# exposed there as anthropic.claude-mythos-preview.
#
# Auth flow at request time:
#   1. boto3 reads ambient credentials (EC2 instance profile in prod,
#      `aws login` session locally)
#   2. STS AssumeRole into the cross-account devs role in Anthropic's
#      preview AWS account (861792231409)
#   3. aws-bedrock-token-generator mints a SigV4-derived bearer token
#      from the assumed credentials
#   4. Bearer token passed to litellm as api_key; litellm routes via
#      its bedrock_mantle provider to the Mantle endpoint
#
# See llm_service._get_bedrock_mythos_api_key() and the bedrock-mythos
# entry in provider_config.PROVIDER_CONFIG.

models = {
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
            "max_tokens": {
                "type": "integer",
                "default": 4096,
                "min": 1,
                "max": 131072,
                "description": "Maximum tokens to generate (up to 128K output).",
            },
        },
    },
}
