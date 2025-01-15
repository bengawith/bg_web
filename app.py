from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json


app = Flask(__name__)

# Secret key for session handling
app.secret_key = os.urandom(24)

# Configure server-side session
app.config['SESSION_TYPE'] = 'file'

# Load the free model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # Replace with your fine-tuned model path after training
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Projects page
@app.route('/projects')
def projects():
    return render_template('projects.html')


# CV page
@app.route('/cv')
def cv():
    return render_template('cv.html')


# Downloadable option for CV
@app.route('/download_cv')
def download_cv():
    return send_from_directory('static', path='CV/Benjamin_Gawith_CV.pdf', as_attachment=True)

# Add distinct pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

with open(r"C:\Users\benga\workspaces\github.com\bg_web\train\ben.json", "r") as file:
    benjamin_data = json.load(file)

def generate_context(user_message):
    """
    Generate dynamic context from ben.json based on the user's message.
    """
    context = f"You are a friendly assistant answering questions about Benjamin Gawith. {benjamin_data['personal_info']['bio']} "

    # Check keywords in the user's message to provide relevant information
    if "project" in user_message.lower():
        projects = "\n".join(
            [f"- {project['title']}: {project['description']}" for project in benjamin_data["projects"]]
        )
        context += f"Here are Benjamin's projects:\n{projects}"
    elif "skill" in user_message.lower():
        skills = ", ".join(benjamin_data["personal_info"]["skills"])
        context += f"His key skills include: {skills}."
    elif "education" in user_message.lower() or "study" in user_message.lower():
        education = "\n".join(
            [f"- {edu['degree']} at {edu['institution']} ({edu['year']})" if "degree" in edu else f"- {edu['qualification']} at {edu['institution']} ({edu['year']})"
             for edu in benjamin_data["personal_info"]["education"]]
        )
        context += f"His education background:\n{education}"
    elif "work" in user_message.lower() or "job" in user_message.lower():
        work_experience = "\n".join(
            [f"- {job['role']} at {job['company']} ({job['date']}): {', '.join(job['responsibilities'])}" for job in benjamin_data["work_experience"]]
        )
        context += f"Here is his work experience:\n{work_experience}"
    elif "volunteer" in user_message.lower():
        volunteering = "\n".join(
            [f"- {volunteer['role']} at {volunteer['organization']}: {volunteer['description']}" for volunteer in benjamin_data["volunteering"]]
        )
        context += f"His volunteering work includes:\n{volunteering}"
    else:
        # Default fallback for questions not covered explicitly
        context += "Please ask about his projects, skills, education, work experience, or volunteering."

    return context


def generate_response(user_message):
    """
    Generate a response by injecting dynamic context.
    """
    context = generate_context(user_message)
    inputs = tokenizer(
        f"{context}\nUser: {user_message}\nAssistant:",
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    reply_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


@app.route('/chat', methods=['POST'])
def chat():
    """
    Flask endpoint to handle chatbot queries.
    """
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'reply': "I didn't receive any message. Please try again!"})

    try:
        # Generate a chatbot response
        bot_reply = generate_response(user_message)
        return jsonify({'reply': bot_reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'reply': "Sorry, something went wrong. Please try again later."})


if __name__ == "__main__":
    app.run(debug=True)