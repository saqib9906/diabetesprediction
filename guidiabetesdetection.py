from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# Add your OpenAI API key here
openai.api_key = "enter your openai api key heer"

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
    return render_template("index.html", response=response)

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
