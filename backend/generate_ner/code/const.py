
model_list = {
    "gpt-3.5-turbo": {"abbr":"tb", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-16k": {"abbr":"tb", "publisher":"openai", "max_tokens":4096*4},
    "gpt-3.5-turbo-0613": {"abbr":"tb0613", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-instruct": {"abbr":"tbinstruct", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-1106": {"abbr":"tb1106", "publisher":"openai", "max_tokens":4096*4},
    "text-davinci-003": {"abbr":"d3", "publisher":"openai", "max_tokens":4097},
    "gpt-4o-mini":{"abbr":"gptmini", "publisher":"openai", "max_tokens":16384},
    # "gemma:3:27b":{"abbr":"gemma327", "publisher":"self", "max_tokens":16384}
}

dataset_language_map = {
    #"wikigold": "en",
    "diffusiondb": "en"
}

my_openai_api_keys = [
    {"key":"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", "set_base":False, "api_base":"YOUR_API_BASE"},
]
