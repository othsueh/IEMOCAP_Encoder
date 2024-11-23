import torch
import numpy as np
import os
import pickle
from tqdm.rich import trange
from transformers import AutoImageProcessor, AutoModel

class VideoLocalEncoder:
    def __init__(self, pretrained_model="facebook/vit-mae-large"):
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model)
    @torch.no_grad()
    def encode_video(self, video):
        inputs = self.image_processor(video, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


def numpy_to_tensor(data):
    # Ensure we have exactly 16 frames by sampling or padding
    target_frames = 16
    total_frames = data.shape[0]
    
    if total_frames >= target_frames:
        # Sample frames evenly
        indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        data = data[indices]
    else:
        # Pad with zeros if we have fewer frames
        padding = np.zeros((target_frames - total_frames, *data.shape[1:]), dtype=data.dtype)
        data = np.concatenate([data, padding], axis=0)
    
    # Convert to tensor and resize to 224x224
    tensor_frames = torch.from_numpy(data).float()
    tensor_frames = tensor_frames / 255.0  # Normalize to [0, 1]
    if torch.cuda.is_available():
        tensor_frames = tensor_frames.cuda()
    
    # Reshape to [B, C, T, H, W] format and resize
    tensor_frames = tensor_frames.permute(0, 3, 1, 2)  # [B, C, H, W]
    tensor_frames = torch.nn.functional.interpolate(
        tensor_frames,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    
    return list(tensor_frames)

def main():
    # Load data
    data_path = '../IEMOCAP/'
    with open(data_path+'data_collected.pickle', 'rb') as file:
        dataset = pickle.load(file)
    print(f"Loaded {len(dataset)} samples")
    
    # create output directory
    output_dir = os.path.join(data_path, "mae_features")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MAE model
    print("Initializing MAE model")
    videoEncoder = VideoLocalEncoder()
    print("=====================================")

    # Process each sample
    for i in trange(len(dataset), desc="Processing samples"):
        sample = dataset[i]
        video = np.load(sample['video']['frames_path'])
        video = numpy_to_tensor(video)
        features = videoEncoder.encode_video(video)
        n_features = features.mean(dim=(0,1))
        avgfeatures = n_features.cpu().numpy()
        
        # Save features
        save_path = os.path.join(output_dir, f"{sample['id']}.npy")
        np.save(save_path, avgfeatures)

if __name__ == "__main__":
    main()  
