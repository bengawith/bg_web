from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import requests
import base64


app = Flask(__name__)

# Secret key for session handling
app.secret_key = os.urandom(24)

# Configure server-side session
app.config['SESSION_TYPE'] = 'file'

# Load the chatbot model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # Change if using a fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GitHub username for project fetching
GITHUB_USERNAME = "bengawith"

# Load local JSON data for chatbot
with open("train/ben.json", "r") as file:
    benjamin_data = json.load(file)

def generate_context(user_message):
    """Generate chatbot context dynamically based on user input."""
    context = f"You are a friendly assistant answering questions about Benjamin Gawith. {benjamin_data['personal_info']['bio']} "

    if "project" in user_message.lower():
        projects = "\n".join(
            [f"- {project['title']}: {project['description']}" for project in benjamin_data["projects"]]
        )
        context += f"Here are Benjamin's projects:\n{projects}"
    elif "skill" in user_message.lower():
        skills = ", ".join(benjamin_data["personal_info"]["skills"])
        context += f"His key skills include: {skills}."
    elif "education" in user_message.lower():
        education = "\n".join(
            [f"- {edu['degree']} at {edu['institution']} ({edu['year']})" for edu in benjamin_data["personal_info"]["education"]]
        )
        context += f"His education background:\n{education}"
    elif "work" in user_message.lower():
        work_experience = "\n".join(
            [f"- {job['role']} at {job['company']} ({job['date']}): {', '.join(job['responsibilities'])}" for job in benjamin_data["work_experience"]]
        )
        context += f"Here is his work experience:\n{work_experience}"
    else:
        context += "Please ask about projects, skills, education, or work experience."

    return context

def generate_response(user_message):
    """Generate a chatbot response based on context."""
    context = generate_context(user_message)
    inputs = tokenizer(
        f"{context}\nUser: {user_message}\nAssistant:",
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    reply_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(reply_ids[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

# Fetch GitHub Projects
@app.route('/projects')
def projects():
    github_api_url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    response = requests.get(github_api_url)
    repos = response.json() if response.status_code == 200 else []
    return render_template('projects.html', repos=repos)


@app.route('/project/<repo_name>/')
@app.route('/project/<repo_name>/<path:sub_path>')
def project_files(repo_name, sub_path=""):
    """Fetch files and directories inside a repository, supporting nested structures."""
    github_api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{sub_path}"
    response = requests.get(github_api_url)
    files = response.json() if response.status_code == 200 else []

    # Generate breadcrumb links
    breadcrumb_parts = sub_path.split('/') if sub_path else []
    breadcrumbs = [{'name': 'Home', 'path': url_for('projects')}]
    
    path_accum = ""
    for part in breadcrumb_parts:
        path_accum += f"{part}/"
        breadcrumbs.append({'name': part, 'path': url_for('project_files', repo_name=repo_name, sub_path=path_accum.rstrip('/'))})

    return render_template('project_files.html', repo_name=repo_name, files=files, breadcrumbs=breadcrumbs, sub_path=sub_path)


@app.route('/project/<repo_name>/file/<path:file_path>')
def view_file(repo_name, file_path):
    github_api_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    response = requests.get(github_api_url)
    file_data = response.json() if response.status_code == 200 else {}

    # Decode the file content
    if "content" in file_data:
        file_data["content"] = base64.b64decode(file_data["content"]).decode("utf-8")

    return render_template('view_file.html', repo_name=repo_name, file=file_data)


@app.route('/cv')
def cv():
    return render_template('cv.html')

@app.route('/view_cv')
def view_cv():
    """Serve the CV PDF for inline viewing."""
    return send_from_directory('static/CV', 'Benjamin_Gawith_CV.pdf')

@app.route('/download_cv')
def download_cv():
    return send_from_directory('static', path='CV/Benjamin_Gawith_CV.pdf', as_attachment=True)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot queries."""
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'reply': "I didn't receive any message. Please try again!"})

    try:
        bot_reply = generate_response(user_message)
        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'reply': "Sorry, something went wrong. Please try again later."})

if __name__ == "__main__":
    app.run(debug=True)
