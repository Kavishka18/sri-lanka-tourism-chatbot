# Sl_Travel_Bot_Chat.py
import random
import json
import torch
import numpy as np
from train_tourism_chatBot import NeuralNet, bag_of_words, tokenize, retrain_model
from database import ChatBotDB
from sentence_transformers import SentenceTransformer, util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('IntentMappings.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

db = ChatBotDB()
bot_name = "Sri Lanka Tourism Bot"

# Initialize sentence transformer for generalized learning
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
similarity_threshold = 0.8  # Threshold for considering questions similar

def get_similar_response(question):
    """Check for similar questions in the database using sentence embeddings."""
    question = question.strip().lower()
    if not question:
        return None
    
    # Get all stored questions
    stored_questions = db.get_all_questions()
    if not stored_questions:
        return None
    
    # Compute embeddings
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    stored_embeddings = sentence_model.encode(stored_questions, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(question_embedding, stored_embeddings)[0]
    
    # Find the most similar question
    max_similarity = similarities.max().item()
    if max_similarity > similarity_threshold:
        max_idx = similarities.argmax().item()
        similar_question = stored_questions[max_idx]
        return db.get_response(similar_question)
    
    return None

def get_response(msg):
    """Generate a response for the user message."""
    global model, all_words, tags, input_size, output_size  # Declare globals at the start

    # Validate input
    if not msg or not msg.strip() or not any(c.isalnum() for c in msg):
        return "Please enter a valid question or message."

    msg = msg.strip().lower()
    try:
        # Check database for exact match
        learned_response = db.get_response(msg)
        if learned_response:
            return learned_response
        
        # Check for similar questions
        similar_response = get_similar_response(msg)
        if similar_response:
            return similar_response  # Return only the actual response
    except RuntimeError as e:
        return f"Sorry, there was a database error: {e}. Please try again."

    try:
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.95:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        return "let me know?"
    except Exception as e:
        return f"Sorry, there was an error processing your message: {e}. Please try again."
    finally:
        # Check if retraining is needed
        new_responses = db.get_retrain_counter()
        if new_responses >= 10:  # Retrain after 10 new responses
            try:
                model, all_words, tags, input_size, output_size = retrain_model(db, device)
                model.eval()
                return "Model retrained with new data! Try your question again."
            except Exception as e:
                return f"Error retraining model: {e}"
            
