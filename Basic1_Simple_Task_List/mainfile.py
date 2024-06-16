import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Function to train the task priority classifier
def train_classifier():
    if not tasks.empty:
        vectorizer = CountVectorizer()
        clf = MultinomialNB()
        model = make_pipeline(vectorizer, clf)
        model.fit(tasks['description'], tasks['priority'])
        return model
    else:
        return None

# Train the model initially if there are existing tasks
model = train_classifier()

# Function to add a task to the list
def add_task(description, priority):
    global tasks  # Declare tasks as a global variable
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    global model  # Update the model with the new data
    model = train_classifier()

# Function to remove a task by description
def remove_task(description):
    global tasks  # Declare tasks as a global variable
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        if model is not None:
            # Get high-priority tasks
            high_priority_tasks = tasks[tasks['priority'] == 'High']

            if not high_priority_tasks.empty:
                # Choose a random high-priority task
                random_task = high_priority_tasks.sample(1).iloc[0]
                print(f"Recommended task: {random_task['description']} - Priority: High")
            else:
                print("No high-priority tasks available for recommendation.")
        else:
            print("No trained model available for recommendations.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        if priority in ['Low', 'Medium', 'High']:
            add_task(description, priority)
            print("Task added successfully.")
        else:
            print("Invalid priority. Please enter Low, Medium, or High.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        if not tasks[tasks['description'] == description].empty:
            remove_task(description)
            print("Task removed successfully.")
        else:
            print("Task not found.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
