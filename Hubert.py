import torch
import numpy as np
import pickle
import os
from transformers import AutoProcessor, HubertModel
from tqdm.rich import trange

class AudioEncoder:
    def __init__(self, model_name="facebook/hubert-large-ls960-ft"):
        self.model = HubertModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode(self, audio_data):
        inputs = self.processor(audio_data, return_tensors="pt", sampling_rate=16000, padding="longest")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state


def numpy_to_tensor(data):
    data = torch.tensor(data).cuda()
    return data

def main():
    # Load data
    data_path = '../IEMOCAP/'
    with open(data_path+'data_collected.pickle', 'rb') as file:
        dataset = pickle.load(file)
    print(f"Loaded {len(dataset)} samples")
    
    # create output directory
    output_dir = os.path.join(data_path, "hubert_features")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MAE model
    print("Initializing Hubert-Large model")
    audioEncoder = AudioEncoder()
    print("=====================================")

    # Process each sample
    for i in trange(len(dataset), desc="Processing samples"):
        sample = dataset[i]
        audio = np.load(sample['audio']['audio_path'])
        audio = numpy_to_tensor(audio)
        features = audioEncoder.encode(audio)
        n_features = features.mean(dim=(0,1))
        avgfeatures = n_features.cpu().numpy()
        
        # Save features
        save_path = os.path.join(output_dir, f"{sample['id']}.npy")
        np.save(save_path, avgfeatures)

if __name__ == "__main__":
    main()  
