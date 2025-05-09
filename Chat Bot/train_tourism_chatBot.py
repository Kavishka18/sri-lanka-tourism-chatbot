# train_tourism_chatBot.py
import numpy as np
import random
import json
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
from database import ChatBotDB

# Download necessary NLTK components
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    """
    Tokenizes a sentence into words.
    """
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    """
    Lemmatizes a word to its base/root form.
    """
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Returns a bag-of-words array:
    1 if the word exists in the sentence, 0 otherwise.
    """
    sentence_words = [lemmatize(word) for word in tokenized_sentence]
    return np.array([1 if w in sentence_words else 0 for w in words], dtype=np.float32)

# NeuralNet class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

def load_intents():
    """Load intents from JSON file."""
    with open('IntentMappings.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_training_data(intents, db):
    """Prepare training data from intents and database."""
    all_words = []
    tags = []
    xy = []

    # Load intents
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Load learned responses from database
    learned_responses = db.get_all_responses()
    for question, answer in learned_responses:
        w = tokenize(question)
        all_words.extend(w)
        tag = f"learned_{question[:50]}"  # Unique tag for each learned question
        tags.append(tag)
        xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = [lemmatize(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    print(len(xy), "patterns")
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    return np.array(X_train), np.array(y_train), all_words, tags

def train_model(X_train, y_train, input_size, hidden_size, output_size, device, num_epochs=1000, batch_size=8, learning_rate=0.001):
    """Train the neural network."""
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')
    return model

def save_model(model, input_size, hidden_size, output_size, all_words, tags):
    """Save the trained model."""
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'training complete. file saved to {FILE}')

def retrain_model(db, device):
    """Retrain the model with new data from the database."""
    intents = load_intents()
    X_train, y_train, all_words, tags = prepare_training_data(intents, db)
    
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    model = train_model(X_train, y_train, input_size, hidden_size, output_size, device)
    save_model(model, input_size, hidden_size, output_size, all_words, tags)
    db.reset_retrain_counter()
    return model, all_words, tags, input_size, output_size

if __name__ == "__main__":
    db = ChatBotDB()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    intents = load_intents()
    X_train, y_train, all_words, tags = prepare_training_data(intents, db)
    
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)
    print(input_size, output_size)

    model = train_model(X_train, y_train, input_size, hidden_size, output_size, device)
    save_model(model, input_size, hidden_size, output_size, all_words, tags)
    db.close_connection()