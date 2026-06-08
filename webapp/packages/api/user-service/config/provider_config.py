from .anthropic import models as anthropic_models
from .bedrock import models as bedrock_models
from .bedrock_mythos import models as bedrock_mythos_models
from .gemini import models as gemini_models
from .openai import models as openai_models
from .openrouter import models as openrouter_models
from .perplexity import models as perplexity_models
PROVIDER_CONFIG = {
    "openai": {
        "api_key_env_var": "OPENAI_API_KEY",
        "models": openai_models
    },
    "gemini": {
        "api_key_env_var": "GEMINI_API_KEY",
        "models":  gemini_models,
    },
    "anthropic": {
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "models": anthropic_models,
    },
    "perplexity": {
        "api_key_env_var": "PERPLEXITYAI_API_KEY",
        "models": perplexity_models,
    },
    "bedrock": {
        "api_key_env_var": "AWS_BEARER_TOKEN_BEDROCK",
        "models": bedrock_models
    },
    # Mythos preview — Bedrock Mantle endpoint in Anthropic's preview
    # AWS account, accessed via cross-account STS AssumeRole + a SigV4-
    # derived bearer token. No static credentials anywhere; the bearer
    # token is minted from short-lived assumed-role credentials at
    # request time, then passed to litellm as api_key. The region is
    # also injected so litellm constructs the right Mantle endpoint
    # URL (bedrock-mantle.{region}.api.aws/v1) — without it litellm
    # defaults to us-east-1.
    #
    # Config knobs:
    #   aws_region              — region where Mythos is provisioned
    #   assume_role_arn         — cross-account devs role to assume
    #   litellm_provider_prefix — built as "bedrock_mantle/<model-id>"
    #                             so litellm routes through its Mantle
    #                             provider (OpenAI-compatible endpoint)
    "bedrock-mythos": {
        "aws_region": "ap-southeast-4",
        "assume_role_arn": "arn:aws:iam::861792231409:role/bedrock-devs",
        "litellm_provider_prefix": "bedrock_mantle",
        "models": bedrock_mythos_models,
    },
    "openrouter": {
        "api_key_env_var": "OPENROUTER_API_KEY",
        "models": openrouter_models,
    },
    "ollama": {
        "models": {
            "llama2": {
                "parameters": {
                    "temperature": {
                        "type": "float",
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Controls randomness"
                    },
                    "num_predict": {
                        "type": "integer",
                        "default": 512,
                        "min": 1,
                        "max": 2048,
                        "description": "Maximum tokens to generate"
                    },
                    "top_p": {
                        "type": "float",
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Nucleus sampling"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 40,
                        "min": 1,
                        "max": 100,
                        "description": "Top-k sampling"
                    },
                    "repeat_penalty": {
                        "type": "float",
                        "default": 1.1,
                        "min": 0.0,
                        "max": 2.0,
                        "description": "Penalty for repetition"
                    }
                }
            },
            "mistral": {
                "parameters": {
                    "temperature": {
                        "type": "float",
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "description": "Controls randomness"
                    },
                    "num_predict": {
                        "type": "integer",
                        "default": 512,
                        "min": 1,
                        "max": 2048,
                        "description": "Maximum tokens to generate"
                    }
                }
            }
        }
    }
}
