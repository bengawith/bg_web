import json

# Data for training the model
data = [
    {"prompt": "Who is Benjamin Gawith?", "completion": "Benjamin Gawith is a software developer and a final-year Mechanical Engineering student at Liverpool John Moores University. He specializes in Python, Flask, machine learning, and web development."},
    {"prompt": "What projects has Benjamin Gawith worked on?", "completion": "Benjamin has worked on projects including a portfolio website, a task logger app, a SpaceWars game, a Mastermind game recreation, a sales summary email app using Flask and Microsoft Azure API, and a neural network for airfoil lift coefficient prediction."},
    {"prompt": "What are Benjamin Gawith's key skills?", "completion": "Benjamin is skilled in Python, Flask, machine learning, web development, communication, time management, problem-solving, teamwork, and IT."},
    {"prompt": "Tell me about Benjamin's Portfolio Website project.", "completion": "Benjamin's Portfolio Website showcases his CV and coding projects in a professional and visually appealing format."},
    {"prompt": "What is the Task Logger App?", "completion": "The Task Logger App is an application that logs tasks, calculates due dates dynamically, and includes features to mark tasks as completed, edit, delete, or add new tasks."},
    {"prompt": "Describe the SpaceWars game project.", "completion": "SpaceWars is a Space Invaders-inspired game built using Pygame, featuring engaging gameplay and custom mechanics."},
    {"prompt": "What is the Mastermind game recreation?", "completion": "The Mastermind game recreation is a digital version of the classic board game developed using Pygame, with interactive gameplay and logic-based challenges."},
    {"prompt": "What is the Sales Summary Email App?", "completion": "The Sales Summary Email App is a Flask web app that uses the Microsoft Azure API to send sales summary emails to vendors based on CSV data."},
    {"prompt": "What is the Aerodynamic Neural Network project?", "completion": "The Aerodynamic Neural Network processes airfoil coordinate files or CST parameters and outputs the lift coefficient (C_L) vs angle of attack (alpha) curves, simplifying and accelerating airfoil design."},
    {"prompt": "What awards has Benjamin Gawith received?", "completion": "Benjamin received the Duke of Edinburgh Award for demonstrating perseverance and organizational skills during an expedition."},
    {"prompt": "What volunteering work has Benjamin Gawith done?", "completion": "Benjamin has volunteered with the Wirral Council Rangers to clean and restore local environments and at Macmillan to organize and clean their shop, showing his passion for community work."}
]

# Save the data to a JSON file
with open("training_data.json", "w") as file:
    json.dump(data, file, indent=4)

print("Training data saved to 'training_data.json'.")

