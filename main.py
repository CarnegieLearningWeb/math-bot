import openai
import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from utils import (N_CHUNKS_TO_CONCAT_BEFORE_UPDATING, OPENAI_API_KEY,
                   SLACK_APP_TOKEN, SLACK_BOT_TOKEN, WAIT_MESSAGE,
                   num_tokens_from_messages, process_conversation_history,
                   update_chat)

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY


def get_conversation_history(channel_id, thread_ts):
    return app.client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        inclusive=True
    )


def convert_arithmetic_expressions(input_str):
    # Remove square brackets and split by comma
    expressions = input_str[1:-1].split(", ")
    output = []
    for exp in expressions:
        exp = exp.strip()
        try:
            # Validate expression with eval()
            result = eval(exp)
            # Format the result based on its type (integer or floating-point)
            if isinstance(result, int):
                formatted_result = str(result)
            elif isinstance(result, float):
                formatted_result = f"{result:.4f}".rstrip("0").rstrip(".")
                if float(formatted_result) != result:
                    formatted_result += "…"
            else:
                # If the result is not a number, skip this expression
                continue
            # Append the expression and its result to the output list
            output.append(f"{exp} = {formatted_result}")
        except:
            # If the expression is invalid or cannot be evaluated, skip it
            pass
    return '\n'.join(output)


def get_altered_system_prompt(expressions):
    equations = convert_arithmetic_expressions(expressions)
    return f'''
You are an AI assistant who can only answer questions about math. Follow these guidelines when answering:

1. Provide step-by-step instructions on how to solve the problem, making use of the provided context of pre-calculated equations. Do not merely give a direct answer to the question. Explain how you arrived at the answer in multiple steps when applicable.
2. If the provided context is insufficient for accurately answering the question or solving the problem, do not perform the calculation. Instead, say, "Sorry, I don't have an answer to that."
3. Do not directly mention the context or refer to it in your response. The user should not be aware that the context was provided. Respond as if you have performed the calculations yourself.

Context:
{equations}
'''


def make_openai_request(messages, channel_id, reply_message_ts):
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.2,
            messages=messages,
            stream=True
        )
        response_text = ""
        ii = 0
        for chunk in openai_response:
            if chunk.choices[0].delta.get("content"):
                ii = ii + 1
                response_text += chunk.choices[0].delta.content
                if response_text.startswith("["):
                    update_chat(app, channel_id, reply_message_ts, "Performing calculation…")
                elif ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                    update_chat(app, channel_id, reply_message_ts, response_text)
                    ii = 0
            elif chunk.choices[0].finish_reason == "stop":
                if response_text.startswith("[") and response_text.endswith("]"):
                    print("expression:\n", response_text)
                    altered_system_prompt = get_altered_system_prompt(response_text)
                    print("altered_system_prompt:\n", altered_system_prompt)
                    messages[0]["content"] = altered_system_prompt
                    update_chat(app, channel_id, reply_message_ts, "Performing calculation…")
                    make_openai_request(messages, channel_id, reply_message_ts)
                else:
                    update_chat(app, channel_id, reply_message_ts, response_text)


@app.event("app_mention")
def command_handler(body, context):
    try:
        channel_id = body["event"]["channel"]
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
        bot_user_id = context["bot_user_id"]
        slack_resp = app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=WAIT_MESSAGE
        )
        reply_message_ts = slack_resp["message"]["ts"]
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


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
