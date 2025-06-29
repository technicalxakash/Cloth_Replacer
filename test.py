import os
import torch
from PIL import Image
from torchvision import transforms

from cp_vton.networks import GMM, TOM  # This is your CP-VTON code
from cp_vton.utils import save_image  # If your utils.py has save_image()

# Paths
INPUT_DIR = './data'
OUTPUT_DIR = './results'
CHECKPOINT_GMM = './checkpoints/gmm_final.pth'
CHECKPOINT_TOM = './checkpoints/tom_final.pth'

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    gmm = GMM().to(device)
    gmm.load_state_dict(torch.load(CHECKPOINT_GMM, map_location=device))
    gmm.eval()

    tom = TOM().to(device)
    tom.load_state_dict(torch.load(CHECKPOINT_TOM, map_location=device))
    tom.eval()

    # Load images
    person = load_image(os.path.join(INPUT_DIR, 'person.jpg')).unsqueeze(0).to(device)
    cloth = load_image(os.path.join(INPUT_DIR, 'cloth.jpg')).unsqueeze(0).to(device)
    cloth_mask = load_image(os.path.join(INPUT_DIR, 'cloth-mask.jpg')).unsqueeze(0).to(device)

    # Run GMM
    with torch.no_grad():
        warped_cloth, _ = gmm(person, cloth)

    # Run TOM
    with torch.no_grad():
        p_rendered, m_composite = tom(person, warped_cloth)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_image(p_tryon, os.path.join(OUTPUT_DIR, 'result.jpg'))

    print('✅ Try-On Image Saved in results/result.jpg')

if __name__ == '__main__':
    main()
