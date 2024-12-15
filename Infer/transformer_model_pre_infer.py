import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from collections import OrderedDict

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model_implement import transformerPre

# Define argument parser
parser = argparse.ArgumentParser(description="Image Captioning Inference")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--model_checkpoint", type=str, default=r"../model/transf_model_pre2.pth", help="Path to model checkpoint")
parser.add_argument("--max_length", type=int, default=50, help="Maximum caption length")


def load_model(model_path, device, tokenizer):

    model = transformerPre.TransformerPre(tokenizer)

    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.eval().to(device)

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

    with torch.no_grad():
        # Encode the input image
        with torch.cuda.amp.autocast():
            generated_ids = model.generate(
                                            image_tensor, 
                                            max_length = max_length, 
                                            num_beams = 4, 
                                            early_stopping=True
                                        )
            caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return caption 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = args.model_checkpoint
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    image_size = 224

    model = load_model(model_checkpoint, device, tokenizer)

    # Preprocess input image
    image_tensor = preprocess_image(args.image_path, image_size).to(device)

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

def main_(image: Image, model_checkpoint: str = r"model/transf_model_pre2.pth", max_length: int = 50):
    """
    Another version of main(), built for web deployment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = model_checkpoint
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    image_size = 224

    model = load_model(model_checkpoint, device, tokenizer)

    # Preprocess input image
    image_tensor = preprocess_image_(image, image_size).to(device)

    # Generate caption
    caption = generate_caption(model, tokenizer, image_tensor, max_length, device)
    return caption

if __name__ == "__main__":
    # args = parser.parse_args()
    # main(args)
    image = Image.open(r'test1.jpg').convert("RGB")
    caption = main_(image)
    print(f"Generated Caption: {caption}")