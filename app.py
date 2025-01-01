from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)


@app.route("/")
def index():
    """
    Render the chat interface using the chat.html template
    """
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """
    Handle the chat interface:
    - accepts POST requests
    - extracts the input from the request
    - passes the input to get_Chat_response to generate a response
    - returns the generated response
    """
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    """
    Generate a chatbot response using Microsoft DialoGPT.

    Args:
        text (str): The input text from the user.

    Returns:
        str: The generated response from the chatbot.

    The function processes the input text over 5 iterations, encoding the input,
    appending it to the chat history, and generating a response using the
    DialoGPT model while keeping the chat history limited to 1000 tokens.
    """

    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(
            str(text) + tokenizer.eos_token, return_tensors="pt"
        )

        # append the new user input tokens to the chat history
        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if step > 0
            else new_user_input_ids
        )

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )

        # pretty print last ouput tokens from bot
        return tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
        )


if __name__ == "__main__":
    app.run()
