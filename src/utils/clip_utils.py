import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import clip
from PIL import Image

def get_clip_image_features(image: torch.Tensor, model) -> torch.Tensor:
    image = image.permute((2, 0, 1)).unsqueeze(0)   
    # clip requires an image of resolution 224
    input_to_resize = (224, 224)
    _transforms =  transforms.Compose([
        transforms.Resize(input_to_resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(input_to_resize),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    image = _transforms(image)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=1, keepdim=True)   

def get_clip_text_features(text: str, model) -> torch.Tensor:
    #TEXT_SEARCH = 'bed'
    device = next(model.parameters()).device
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features    