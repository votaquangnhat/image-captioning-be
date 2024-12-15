import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torch.distributions import Categorical
from transformers import AutoTokenizer

from collections import OrderedDict

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model_implement.transformer import EncoderDecoder  # Your custom model definition

# Define argument parser
parser = argparse.ArgumentParser(description="Image Captioning Inference")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--model_checkpoint", type=str, default=r"../model/transf_model.pth", help="Path to model checkpoint")
parser.add_argument("--max_length", type=int, default=50, help="Maximum caption length")


def load_model(model_path, device, image_size, val_images, tokenizer):
    """Load the model and set to evaluation mode."""
    model = EncoderDecoder(image_size=image_size, channels_in=val_images.shape[1],
                                     num_emb=tokenizer.vocab_size, patch_size=8,
                                     num_layers=(6, 6), hidden_size=192,
                                     num_heads=8)

    state_dict = torch.load(model_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path, image_size):
    """Preprocess the input image."""
    transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_caption(model, tokenizer, image_tensor, max_length, device):
    """Generate captions from the model."""

    # Add the Start-Of-Sentence token to the prompt to signal the network to start generating the caption
    sos_token = 101 * torch.ones(1, 1).long()

    # Set the temperature for sampling during generation
    temp = 0.5

    log_tokens = [sos_token]

    with torch.no_grad():
        # Encode the input image
        with torch.cuda.amp.autocast():
            # Forward pass
            image_embedding = model.encoder(image_tensor.to(device))

        # Generate the answer tokens
        for _ in range(max_length):
            input_tokens = torch.cat(log_tokens, 1)

            # Decode the input tokens into the next predicted tokens
            data_pred = model.decoder(input_tokens.to(device), image_embedding)

            # Sample from the distribution of predicted probabilities
            dist = Categorical(logits=data_pred[:, -1] / temp)
            next_tokens = dist.sample().reshape(1, 1)

            # Append the next predicted token to the sequence
            log_tokens.append(next_tokens.cpu())

            # Break the loop if the End-Of-Caption token is predicted
            if next_tokens.item() == 102:
                break

    # Convert the list of token indices to a tensor
    pred_text = torch.cat(log_tokens, 1)

    # Convert the token indices to their corresponding strings using the vocabulary
    pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)

    # Join the token strings to form the predicted text
    pred_text = "".join(pred_text_strings)
    
    return pred_text  

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    image_size = 128

    # Preprocess input image
    image_tensor = preprocess_image(args.image_path, image_size).to(device)

    model = load_model(model_checkpoint, device,image_size, image_tensor, tokenizer)

    # Generate caption
    caption = generate_caption(model, tokenizer, image_tensor, args.max_length, device)
    print(f"Generated Caption: {caption}")

def preprocess_image_(image: Image, image_size):
    """Another version of preprocess_image(), built for main_()."""
    transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0)  # Add batch dimension

def main_(image: Image, model_checkpoint: str = r"model/transf_model.pth", max_length: int = 50):
    """
    Another version of main(), built for web deployment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    image_size = 128

    # Preprocess input image
    image_tensor = preprocess_image_(image, image_size).to(device)

    model = load_model(model_checkpoint, device,image_size, image_tensor, tokenizer)

    # Generate caption
    caption = generate_caption(model, tokenizer, image_tensor, max_length, device)
    return caption

if __name__ == "__main__":
    # args = parser.parse_args()
    # main(args)
    image = Image.open(r'test1.jpg').convert("RGB")
    caption = main_(image)
    print(f"Generated Caption: {caption}")