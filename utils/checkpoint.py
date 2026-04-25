import pickle
import os
import json

def save(transformer, epoch):
    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(transformer, f)

def load(epoch):
    with open(f'checkpoints/epoch_{epoch}.pkl', 'rb') as f:
        return pickle.load(f)

def save_losses(losses, filename='loss_history.json'):
    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{filename}', 'w') as f:
        json.dump(losses, f)