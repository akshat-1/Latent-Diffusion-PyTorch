import src.model_loader as model_loader
import src.pipeline as pipeline
import cv2
from transformers import CLIPTokenizer
import torch

DEVICE= "cpu"

ALLOW_CUDA = True
ALLOW_MPS= True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE= "mps"

print(f"Using device {DEVICE}")

tokenizer = CLIPTokenizer("data/vocab.json" , merges_file = "data/merges.txt")
model_file = "data/v1-5-pruned-emaonly.ckpt"
models = model_loader.perload_models_from_standard_weights(model_file, DEVICE)

prompt = "Generate an image with text - Envisage 14.0 Summer Training in white background, with lots of designs in 4k"

uncond_prompt = "Dark background"  
do_cfg = True
cfg_scale =7


#I - I 

input_image = None
image_path = "images/temp.png"
if(input_image): 
    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    

strength = 0.9
sampler ="ddpm"
num_inference_steps = 100
seed =42

output_image = pipeline.generate(
    prompt = prompt,
    uncond_prompt= uncond_prompt,
    input_image=input_image,
    strength= strength,
    do_cfg = do_cfg,
    cfg_scale = cfg_scale,
    sampler_name = sampler,
    n_inference_steps = num_inference_steps,
    seed =seed,
    models = models,
    device = DEVICE,
    idle_device = "cpu",
    tokenizer = tokenizer


)
cv2.imwrite('output_image.jpg', output_image)
cv2.imshow('Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()