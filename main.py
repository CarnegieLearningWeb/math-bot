import openai
import os
from dotenv import load_dotenv
from enum import Enum
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from utils import (N_CHUNKS_TO_CONCAT_BEFORE_UPDATING, OPENAI_API_KEY, MAX_TOKENS,
                   SLACK_APP_TOKEN, SLACK_BOT_TOKEN, WAIT_MESSAGE, SYSTEM_PROMPT,
                   num_tokens_from_messages, process_conversation_history,
                   update_chat)

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY

class Category(Enum):
    UNDEFINED = 0
    CALCULATION_BASED = 1
    CONCEPTUAL_INFORMATIONAL = 2
    MATH_PROBLEM_GENERATION = 3
    GREETINGS_SOCIAL = 4
    OFF_TOPIC = 5
    MISCELLANEOUS = 6

MAX_FAILED_ATTEMPTS = 3
FAILED_ATTEMPT_MESSAGE = "Apologies, there was an unexpected issue. We're attempting to process your request again..."
ERROR_MESSAGE = "We're sorry, there was an error processing your request. Please report this issue to zlee@carnegielearning.com."
MAX_TOKEN_MESSAGE = f"Apologies, but the maximum number of tokens ({format(MAX_TOKENS, ',')}) for this thread has been reached. Please start a new thread to continue discussing this topic."


def get_conversation_history(channel_id, thread_ts):
    return app.client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        inclusive=True
    )


def get_altered_system_prompt(system_prompt_category, is_calculated=False, equations=""):
    altered_system_prompt = ""
    if system_prompt_category == Category.CALCULATION_BASED.value:
        if is_calculated is False:
            altered_system_prompt = """
You are a researcher tasked with identifying where calculation is needed in the user input. Please follow these guidelines when answering:
1. Do not perform the calculation. Instead, your entire response should be a list of the arithmetic expressions that need to be calculated step by step, separated by commas, and wrap them with square brackets, like the examples below:

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

Q: What is 2 to the power of 3?
[2 ** 3]

2. Do not include expressions that cannot be calculated (e.g., algebraic expressions) because these expressions will be parsed and converted to equations by a Python function.
3. For the same reason, avoid using mathematical constants or symbols, such as π or e, in the arithmetic expressions. Only use numbers and basic arithmetic operations that Python can interpret with the eval function. Convert mathematical constants or symbols to numbers when applicable.

Only consider the latest user input to identify where calculation is needed.
Again, when you provide the list of arithmetic expressions, that should be the entire response, and nothing else should be included in the response. If the list is empty, your response should be [].
Let's work this out in a step by step way to be sure we have the right list of expressions.
"""
        else:
            altered_system_prompt = f"""
You are MathBot, a K-12 math tutor chatbot tasked with guiding students through math questions. Please follow these guidelines when answering:

1. Provide step by step instructions on how to solve the problem, making use of the provided context of pre-calculated equations. Rather than giving the direct answer, demonstrate how you arrived at the answer through multiple steps.
2. If the provided context is insufficient for accurately answering the question or solving the problem, respond with, "Sorry, I don't have enough information to solve that."
3. Incorporate the context naturally in your responses without explicitly mentioning it. Make your responses seem as if you've performed the calculations yourself.

Let's work this out in a step by step way to be sure we have the right instructions for the student.

Context:
{equations}
"""
    elif system_prompt_category == Category.CONCEPTUAL_INFORMATIONAL.value:
        altered_system_prompt = """
You are MathBot, a K-12 math tutor chatbot tasked with guiding students through math questions. Please follow these guidelines when answering:

1. Explain the reason for your answer, or how you arrived at the answer in multiple steps when applicable.
2. Provide the base knowledge for students to better understand your answer when applicable.
3. If the question is incomplete, unclear to answer, or unsolvable, ask for clarification or explain why you cannot answer it.

Let's work this out in a step by step way to be sure we have the right instructions for the student.
"""
    elif system_prompt_category == Category.MATH_PROBLEM_GENERATION.value:
        altered_system_prompt = """
You are a researcher tasked with generating math problems as requested in the user input. Follow these guidelines when generating:

1. When generating math problems, ensure they can be explained and solved in a step-by-step manner. Adjust the difficulty of the problem according to the user's age or grade level if provided.
2. For word problems, use language that is clear, easy to understand, and safe for K-12 students.
3. If the user's request for a math problem is unclear or lacks necessary information, ask for clarification or provide an explanation of why the problem cannot be generated.

Let's work this out in a step by step way to be sure we have the right problems for the student.
"""
    elif system_prompt_category == Category.GREETINGS_SOCIAL.value:
        altered_system_prompt = """
You are MathBot, a K-12 math tutor chatbot tasked with guiding students through math questions. Please follow these guidelines when answering:

1. Maintain a friendly and engaging conversation with students.
2. Encourage and motivate students to foster a positive attitude towards math when appropriate.
3. Ensure all responses are safe and appropriate for K-12 students, and gently guide students to use respectful and appropriate language if necessary.

Let's work this out in a step by step way to be sure we have the right instructions for the student.
"""
    elif system_prompt_category == Category.OFF_TOPIC.value:
        altered_system_prompt = """
You are MathBot, a K-12 math tutor chatbot tasked with guiding students through math questions. Please follow these guidelines when answering:

1. If the user input is not related to math, say "I'm here to help with math-related questions. Can we focus on a math problem or concept?"
2. If the user input is related to math but not directly (e.g., coding-related questions), say "My expertise is in math topics. Could we focus on a question that's directly related to math?"
3. If the user input is incomplete or unclear to answer, ask for clarification or explain why you cannot answer it.

Let's work this out in a step by step way to be sure we have the right instructions for the student.
"""
    elif system_prompt_category == Category.MISCELLANEOUS.value:
        altered_system_prompt = """
You are MathBot, a K-12 math tutor chatbot tasked with guiding students through math questions. Please follow these guidelines when answering:

1. If the user input is an emotional interjection, empathize and provide supportive responses when appropriate.
2. If the user input is gibberish (i.e., nonsensical or random characters), gently notify the user that the input was not understood and ask for a clear question or statement.
3. If the user input is unclear or ambiguous but still appears to be an attempt at a meaningful question or statement, ask for clarification.

Let's work this out in a step by step way to be sure we have the right instructions for the student.
"""
    return altered_system_prompt


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
    return "\n".join(output)


def make_openai_request(messages, channel_id, reply_message_ts, system_prompt_category=Category.UNDEFINED.value,
                        is_calculated=False, num_failed_attempts=0):
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
            if system_prompt_category == Category.CALCULATION_BASED.value and is_calculated is False and response_text.startswith("["):
                update_chat(app, channel_id, reply_message_ts, f"Performing calculation{'.' * ((ii % 3) + 1)}")
            elif ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                update_chat(app, channel_id, reply_message_ts, response_text)
                ii = 0
        elif chunk.choices[0].finish_reason == "stop":
            if system_prompt_category == Category.UNDEFINED.value:
                if response_text.isdigit():
                    system_prompt_category = int(response_text)
                    altered_system_prompt = get_altered_system_prompt(system_prompt_category)
                    if altered_system_prompt:
                        messages[0]["content"] = altered_system_prompt
                        return make_openai_request(messages, channel_id, reply_message_ts, system_prompt_category,
                                                    is_calculated=False, num_failed_attempts=num_failed_attempts)
            elif system_prompt_category == Category.CALCULATION_BASED.value and is_calculated is False:
                if response_text.startswith("[") and response_text.endswith("]"):
                    equations = convert_arithmetic_expressions(response_text)
                    altered_system_prompt = get_altered_system_prompt(system_prompt_category, is_calculated=True, equations=equations)
                    if altered_system_prompt:
                        messages[0]["content"] = altered_system_prompt
                        update_chat(app, channel_id, reply_message_ts, f"Performing calculation{'.' * ((ii % 3) + 1)}")
                        return make_openai_request(messages, channel_id, reply_message_ts, system_prompt_category,
                                                    is_calculated=True, num_failed_attempts=num_failed_attempts)
            else:
                return update_chat(app, channel_id, reply_message_ts, response_text)
            
            # If something unexpected happens, start over using the original system prompt
            if num_failed_attempts < MAX_FAILED_ATTEMPTS:
                update_chat(app, channel_id, reply_message_ts, FAILED_ATTEMPT_MESSAGE)
                messages[0]["content"] = SYSTEM_PROMPT
                return make_openai_request(messages[:1] + messages[num_failed_attempts - 3:] if len(messages) > 4 - num_failed_attempts else messages[:], 
                                            channel_id, reply_message_ts, system_prompt_category=Category.UNDEFINED.value,
                                            is_calculated=False, num_failed_attempts=num_failed_attempts+1)
            
            # If maximum failed attempts was reached, respond with an error message
            update_chat(app, channel_id, reply_message_ts, ERROR_MESSAGE)
        elif chunk.choices[0].finish_reason == "length":
            update_chat(app, channel_id, reply_message_ts, response_text + "...\n\n" + MAX_TOKEN_MESSAGE)


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
