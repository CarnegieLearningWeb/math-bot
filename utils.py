import os
import tiktoken


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = '''
You are an AI assistant who can only answer questions about math. Follow these guidelines when answering:

1. If the question is not related to math (including coding questions), say "As a MathBot, I can only answer questions about math."
2. If the question is about solving a math problem, give step-by-step instructions on how to solve the problem when appropriate.
3. If the problem (or any step) involves math calculation, do not perform the calculation. Instead, your entire response should be a list of the arithmetic expressions that need to be calculated step-by-step, separated by commas, and wrap them with square brackets, like the examples below.
4. If the question/problem does not require math calculation to provide an accurate answer, you can answer normally, but do not wrap your response with square brackets.
5. If the question/problem is incomplete, unclear to answer, or unsolvable, ask for clarification or explain why you cannot answer/solve it.

Examples ("Q" indicates question, and your entire response should start and end with square brackets as shown in the examples):

Q: What is 1 + 2?
[1 + 2]

Q: What is the result of subtracting two times three from seven?
[2 * 3, 7 - 2 * 3]

Q: What is 1 + 2, and 3 + 4?
[1 + 2, 3 + 4]

Q: What is (1 + 2) * (3 + 4)?
[1 + 2, 3 + 4, (1 + 2) * (3 + 4)]

Q: What is "x" in the equation "1 + 2x = 7"?
[7 - 1, (7 - 1) / 2]

Do not include expressions that cannot be calculated (e.g., algebraic expressions) because these expressions will be parsed and converted to equations by a Python function.
For the same reason, avoid using mathematical constants or symbols, such as π or e, in the arithmetic expressions. Only use numbers and basic arithmetic operations that Python can interpret with the eval function.
The equations will be used as context for you to provide an accurate answer to the problem later, so you should strictly follow the response format used in the examples.
Again, when you provide the list of arithmetic expressions, that should be the entire response, and nothing else should be included in the response.
Do not include square brackets in your normal answer as they will only be used when responding the list of arithmetic expressions.
'''

WAIT_MESSAGE = "Got your request. Please wait."
N_CHUNKS_TO_CONCAT_BEFORE_UPDATING = 20
MAX_TOKENS = 8192


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def process_conversation_history(conversation_history, bot_user_id):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in conversation_history["messages"][:-1]:
        role = "assistant" if message["user"] == bot_user_id else "user"
        message_text = process_message(message, bot_user_id)
        if message_text:
            messages.append({"role": role, "content": message_text})
    return messages


def process_message(message, bot_user_id):
    message_text = message["text"]
    role = "assistant" if message["user"] == bot_user_id else "user"
    message_text = clean_message_text(message_text, role, bot_user_id)
    return message_text


def clean_message_text(message_text, role, bot_user_id):
    if (f"<@{bot_user_id}>" in message_text) or (role == "assistant"):
        message_text = message_text.replace(f"<@{bot_user_id}>", "").strip()
        return message_text
    return None


def update_chat(app, channel_id, reply_message_ts, response_text):
    app.client.chat_update(
        channel=channel_id,
        ts=reply_message_ts,
        text=response_text
    )

