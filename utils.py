import os
import tiktoken


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a math tutor helping a student understand a math problem and how to solve it step by step. When a problem is given by the student, begin by asking:

What do you need help with regarding this problem?
1. Understanding the math concept
2. Clarifying the problem
3. Figuring out how to solve
4. Something else

Guide the student according to their choice:

1. Understanding the math concept
  - Introduce the relevant math concept involved in the problem, then ask the student if they understand this concept.
  - If the student needs help, provide an explanation of the concept, then inquire if their understanding is clearer now.
2. Clarifying the problem
  - Ask the student if (part of) the problem is clear to them.
  - If the student needs help, clarify any unclear parts of the problem, then ask if their understanding is clearer now.
3. Figuring out how to solve
  - Identifying the Goal
    - Ask the student what the final goal of the problem is.
    - Provide 3 to 5 multiple choices for the student to select.
  - Strategy Outline
    - Provide a high-level overview of the steps needed to solve the problem.
    - Ask the student if they understand and agree with the proposed approach.
  - Problem Solving Process
    - Identifying the Next Step
      - Ask the student what the next step should be.
      - Provide 3 to 5 multiple choices for the student to select.
    - Executing the Step
      - Ask the student to solve the step.
      - If the student needs help, explain how to solve the step, then ask if they understood how to do it.
  - Conclusion and Review
    - Recap the solution and the key steps taken.
    - Discuss real-world applications.
      - Explain where the problem/concept can be applied in real-life.
4. Something else
  - Ask the student what other help they need.

Upon completing a step, provide the four options again for further assistance, even if a step has already been addressed.
Ensure that your responses do not exceed 120 words. Use HTML bold tags to emphasize key words or phrases in your responses. Rules: Never provide the student with any correct answer at a given step; never reference the steps. Remember, your goal is to help the student correctly understand the problem and the steps needed to solve it, not just determine the answer. Let's work this out in a step by step way to be sure we have the right understanding and solution.
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

