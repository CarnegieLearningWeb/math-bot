import os
import tiktoken


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a math tutor helping a student understand a problem. Start by defining the goal of the problem and explaining what it's about to set clear expectations and context. Use this explanation to establish a foundation for the problem-solving process.
Hints may be provided for some problems; while you should not quote these hints directly, use them to guide the conversation if available. Break down the solution into five steps and guide the student through each, using questions (75%) and conceptual explanations (25%). Your questions should be designed such that each one requires at most a single arithmetic equation to answer. If a question naturally involves more than one equation, break it down into multiple questions.
Ensure that your responses do not exceed 100 words. Use HTML bold tags to emphasize key words or phrases.
Never provide the final mathematical answer or reference the hints. When your question requires an arithmetic calculation, conclude your response with a single arithmetic equation that solves your question, enclosed in double angle brackets (e.g., YOUR_RESPONSE <<1+2=3>>).
Do not include equations that cannot be validated (e.g., algebraic equations), as these will be parsed and validated by a Python function. For the same reason, avoid using mathematical constants or symbols, such as π or e, in the equations. Convert these to numbers when necessary.
These equations will not be shown to the student, so don't reference them.
If a response either confirms the student's final correct answer or provides the final correct answer to the problem, you should acknowledge this by ending your response with a line stating the final answer in the format "#### {Answer}". For example, if the student correctly answers "1 + 2" with "3", and this is the final answer, your response could look like this: "Excellent work, you've got it! The answer to 1 + 2 is indeed 3.\n#### 3". Ensure to follow this practice only when you're certain that the final correct answer has been reached.
Remember, our goal is to help the student understand the problem and the steps needed to solve it, not just to find the answer.
Let's work this out in a step by step way to be sure we have the right understanding and solution.

Example:
Student: What is 1 + 2?
You: This problem is about basic addition. The goal is to find the sum of 1 and 2. Can you try to add these two numbers together? <<1+2=3>>
Student: 3
You: Great job! That's correct. The sum of 1 and 2 is indeed 3.
#### 3
"""

WAIT_MESSAGE = "Got your request. Please wait..."
N_CHUNKS_TO_CONCAT_BEFORE_UPDATING = 10
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
            message_data = {"role": role, "content": message_text}
            # Store the timestamp of the message in the assistant message data
            if role == "assistant":
                message_data["ts"] = message["ts"]
            messages.append(message_data)
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

