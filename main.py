import math
import os
import re
import openai
from threading import Event
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from utils import (N_CHUNKS_TO_CONCAT_BEFORE_UPDATING, OPENAI_API_KEY, MAX_TOKENS,
                   SLACK_APP_TOKEN, SLACK_BOT_TOKEN, WAIT_MESSAGE,
                   num_tokens_from_messages, process_conversation_history,
                   update_chat)

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY

# Debug mode
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

MAX_TOKEN_MESSAGE = f"Apologies, but the maximum number of tokens ({format(MAX_TOKENS, ',')}) for this thread has been reached. Please start a new thread to continue discussing this topic."

# A dictionary to store an event for each channel
events = {}

# keys: ts (message timestamp), values: equation string
equation_dict = {}


def get_conversation_history(channel_id, thread_ts):
    return app.client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        inclusive=True
    )


def process_equation(equation):
    # Split the equation into left and right parts
    parts = equation.split("=")
    if len(parts) != 2:  # If equation can't be split into exactly two parts, return as is
        return equation

    # Clean whitespaces
    left, right = parts[0].replace(" ", ""), parts[1].strip()

    # Check if left is a number, or if right is not a number
    if re.fullmatch(r"^-?\d+(\.\d+)?$", left) or not re.fullmatch(r"^-?\d+(\.\d+)?$", right):
        return equation

    # Create a safe environment for eval and try to evaluate the left part
    safe_env = dict(__builtins__=None, math=math)
    try:
        result = eval(left, safe_env)
    except (NameError, SyntaxError, TypeError):
        return equation

    # Format the result based on its type
    if isinstance(result, int):
        return f"{left}={result}"
    elif isinstance(result, float):
        truncated_result = int(result * 10000) / 10000
        return f"{left}={truncated_result:.4f}…" if result != truncated_result else f"{left}={truncated_result}"


def make_openai_request(messages, channel_id, reply_message_ts):
    altered_messages = []

    for message in messages:
        altered_messages.append({"role": message["role"], "content": message["content"]})
        ts = message.get("ts")
        if ts and equation_dict.get(ts) is not None:
            altered_messages[-1]["content"] += f" <<{equation_dict[ts]}>>"

    debug_print("Altered Messages:")
    for msg in altered_messages[1:]:
        debug_print(f'• {msg["role"].title()}: {msg["content"]}')

    openai_response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.4,
        messages=altered_messages,
        stream=True
    )
    response_text = ""
    is_parsing = False
    equation = ""
    ii = 0
    for chunk in openai_response:
        if chunk.choices[0].delta.get("content"):
            ii = ii + 1
            token = chunk.choices[0].delta.content

            # Parse and process the equation (hidden in the chat)
            if not is_parsing:
                if not token.endswith("<<"):
                    response_text += token
                else:
                    is_parsing = True
            else:
                if not token.endswith(">>"):
                    equation += token
                else:
                    equation = process_equation(equation)
                    debug_print(f"Processed Equation: {equation}")
                    equation_dict[reply_message_ts] = equation
                    equation = ""
                    is_parsing = False
                                                    
            if ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                update_chat(app, channel_id, reply_message_ts, response_text)
                ii = 0
        elif chunk.choices[0].finish_reason == "stop":
            update_chat(app, channel_id, reply_message_ts, response_text)
        elif chunk.choices[0].finish_reason == "length":
            update_chat(app, channel_id, reply_message_ts, response_text + "...\n\n" + MAX_TOKEN_MESSAGE)


@app.event("app_mention")
def command_handler(body, context):
    channel_id = body["event"]["channel"]
    thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
    bot_user_id = context["bot_user_id"]
    slack_resp = app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=WAIT_MESSAGE
    )
    reply_message_ts = slack_resp["message"]["ts"]
    
    # If there's no event for this thread yet, create one and set it
    if thread_ts not in events:
        events[thread_ts] = Event()
        events[thread_ts].set()

    # Wait until the event is set, indicating that the previous message is done processing
    events[thread_ts].wait()

    # Clear the event to indicate that a new message is being processed
    events[thread_ts].clear()

    try:
        conversation_history = get_conversation_history(channel_id, thread_ts)
        messages = process_conversation_history(conversation_history, bot_user_id)
        num_tokens = num_tokens_from_messages(messages)
        print(f"Number of tokens: {num_tokens}")
        make_openai_request(messages, channel_id, reply_message_ts)
    except Exception as e:
        print(f"Error: {e}")
        app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"I can't provide a response. Encountered an error:\n`\n{e}\n`")
    finally:
        # Set the event to indicate that this message is done processing
        events[thread_ts].set()


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
