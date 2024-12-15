from dumbModel.modules import FeatureExtractor,Encoder,Decoder,Config

from PIL import Image
import json
import torchvision
import torch

def greedy_decode(config:Config,encoder:Encoder,decoder:Decoder,feature_extractor:FeatureExtractor,idx2word:dict,word2idx:dict,device,image:torch.Tensor):
    """
    Function that performs greedy decoding on the given image.

    Arguments:
        config: Configuration class
        encoder (Encoder): Encoder class.
        decoder (Decoder): Decoder class.
        feature_extractor (FeatureExtractor): FeatureExtractor class.
        idx2word (dict): text representations of the indexes.
        word2idx (dict): index representations of the texts.
        device (torch.device): Device. (whether 'cuda' or 'cpu')
        image (torch.Tensor): Tensor-based image.

    Returns:
        caption (str): Caption decoded by decoder.
    """

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    image_features = feature_extractor(image)
    encoder_outputs = encoder(image_features)

    words = torch.Tensor([word2idx['<start>']] + [word2idx['<pad>']] * (config.maxlen-1)).to(device).long().unsqueeze(0)
    pad = torch.Tensor([True] * config.maxlen).to(device).bool().unsqueeze(0)
    generated_caption = []
    for i in range(config.maxlen -1):
        pad[:,i] = False
        y_pred_prob = decoder(x=words,encoder_outputs=encoder_outputs,tgt_key_padding_mask=pad)
        y_pred_prob = y_pred_prob[:,i].clone()
        y_pred = y_pred_prob.argmax(-1)
        generated_caption.append(idx2word[y_pred[0].item()])
        if y_pred[0] == word2idx['<end>']:
            break

        if i < (config.maxlen-1):
            words[:,i+1] = y_pred.view(-1)

    generated_caption.remove("<end>")

    caption = " ".join(generated_caption)

    return caption

def infer(image):
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    with open("dumbModel/word2idx.json","r") as json_file:
        word2idx = json.load(json_file)
    config.vocab_size = len(word2idx)
    encoder = Encoder(config).to(device)
    feat_ext = FeatureExtractor().to(device)
    decoder = Decoder(config).to(device)
    decoder.load_state_dict(torch.load("dumbModel/decoder.pth", map_location=torch.device('cpu'), weights_only=True))
    encoder.load_state_dict(torch.load("dumbModel/encoder.pth", map_location=torch.device('cpu'), weights_only=True))

    decoder.eval()
    encoder.eval()

    idx2word = {v:k for k,v in word2idx.items()}
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()
        ]
    )
    image = transforms(image)
    with torch.no_grad():
        caption = greedy_decode(image=image,config=config,word2idx=word2idx,decoder=decoder,device=device,encoder=encoder,feature_extractor=feat_ext,idx2word=idx2word)
    
    return caption

def evaluation():
    pass

if __name__ == "__main__":
    image_path = r"test3.jpg"
    image = Image.open(image_path)
    caption = infer(image)
    