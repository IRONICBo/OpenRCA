import os
import yaml
import time

def load_config(config_path=None):
    if config_path is None:
        # Allow per-worker config via environment variable
        env_config = os.environ.get("RCA_API_CONFIG")
        if env_config and os.path.exists(env_config):
            config_path = env_config
        else:
            # Search for config in multiple locations
            candidates = [
                "rca/api_config.yaml",
                os.path.join(os.path.dirname(__file__), "api_config.yaml"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    config_path = c
                    break
        if config_path is None:
            raise FileNotFoundError("Cannot find api_config.yaml")

    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs

configs = load_config()

def OpenAI_chat_completion(messages, temperature):
    from openai import OpenAI
    client = OpenAI(
        api_key=configs["API_KEY"]
    )
    return client.chat.completions.create(
        model = configs["MODEL"],
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def Google_chat_completion(messages, temperature):
    import google.generativeai as genai
    genai.configure(
        api_key=configs["API_KEY"]
    )
    genai.GenerationConfig(temperature=temperature)
    system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None
    messages = [item for item in messages if item["role"] != "system"]
    messages = [{"role": "model" if item["role"] == "assistant" else item["role"], "parts": item["content"]} for item in messages]
    history = messages[:-1]
    message = messages[-1]
    return genai.GenerativeModel(
        model_name=configs["MODEL"],
        system_instruction=system_instruction
        ).start_chat(
            history=history if history != [] else None
            ).send_message(message).text

def Anthropic_chat_completion(messages, temperature):
    import anthropic
    client = anthropic.Anthropic(
        api_key=configs["API_KEY"]
    )
    # Extract system prompt from messages (Anthropic API requires separate system parameter)
    system_prompt = None
    filtered_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            filtered_messages.append(msg)

    # Ensure messages alternate between user and assistant
    # Merge consecutive same-role messages if needed
    merged_messages = []
    for msg in filtered_messages:
        if merged_messages and merged_messages[-1]["role"] == msg["role"]:
            merged_messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged_messages.append({"role": msg["role"], "content": msg["content"]})

    max_tokens = int(configs.get("MAX_TOKENS", 8192))

    kwargs = {
        "model": configs["MODEL"],
        "messages": merged_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    # Handle new Anthropic API response format
    if hasattr(response, 'content') and isinstance(response.content, list):
        return "".join([block.text for block in response.content if hasattr(block, 'text')])
    return response.content

# for 3-rd party API which is compatible with OpenAI API (with different 'API_BASE')
def AI_chat_completion(messages, temperature):
    from openai import OpenAI
    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"]
    )
    return client.chat.completions.create(
        model = configs["MODEL"],
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content


def WanQing_chat_completion(messages, temperature):
    """Chat completion via WanQing internal API gateway (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=configs.get("API_KEY") or os.environ.get("WQ_API_KEY"),
        base_url=configs.get("API_BASE", "http://wanqing.internal/api/gateway/v1/endpoints"),
    )

    max_tokens = int(configs.get("MAX_TOKENS", 8192))

    if temperature == 0.0:
        temperature = 0.01

    kwargs = {
        "model": configs["MODEL"],
        "messages": messages,
    }

    # Some models (e.g. GPT-5) don't support temperature/max_tokens params
    if not configs.get("NO_TEMPERATURE"):
        kwargs["temperature"] = temperature
    if not configs.get("NO_MAX_TOKENS"):
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    if content is None:
        finish_reason = response.choices[0].finish_reason
        raise ValueError(f"WanQing API returned empty content (finish_reason={finish_reason})")
    return content


def vLLM_chat_completion(messages, temperature):
    """Chat completion for models served via vLLM (OpenAI-compatible API)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=configs["API_KEY"],
        base_url=configs["API_BASE"],
    )

    max_tokens = int(configs.get("MAX_TOKENS", 8192))

    # vLLM: temperature=0.0 may cause issues on some versions, use a tiny value
    if temperature == 0.0:
        temperature = 0.01

    response = client.chat.completions.create(
        model=configs["MODEL"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    if content is None:
        # vLLM may return None content on truncation or empty generation
        finish_reason = response.choices[0].finish_reason
        raise ValueError(
            f"vLLM returned empty content (finish_reason={finish_reason}). "
            "Model may have hit max_tokens or failed to generate."
        )
    return content


def get_chat_completion(messages, temperature=0.0):

    def send_request():
        if configs["SOURCE"] == "vLLM":
            return vLLM_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "WanQing":
            return WanQing_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "AI":
            return AI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "OpenAI":
            return OpenAI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Google":
            return Google_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Anthropic":
            return Anthropic_chat_completion(messages, temperature)
        else:
            raise ValueError(f"Invalid SOURCE in api_config: '{configs['SOURCE']}'. "
                             "Use 'vLLM', 'WanQing', 'AI', 'OpenAI', 'Google', or 'Anthropic'.")

    max_retries = int(configs.get("MAX_RETRIES", 3))
    for i in range(max_retries):
        try:
            return send_request()
        except Exception as e:
            print(f"API call attempt {i+1}/{max_retries} failed: {e}")
            if '429' in str(e) or 'rate' in str(e).lower():
                wait_time = min(2 ** i, 30)
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                continue
            elif '529' in str(e) or 'overloaded' in str(e).lower():
                wait_time = min(5 * (i + 1), 60)
                print(f"API overloaded. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                continue
            elif 'Connection' in str(type(e).__name__) or 'connection' in str(e).lower():
                wait_time = min(3 * (i + 1), 30)
                print(f"Connection error to vLLM server. Waiting {wait_time}s. "
                      f"Check that vLLM is running at {configs.get('API_BASE', 'unknown')}")
                time.sleep(wait_time)
                continue
            else:
                if i == max_retries - 1:
                    raise e
                time.sleep(1)
