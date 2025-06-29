import torchvision.utils as vutils

def save_image(tensor, path):
    vutils.save_image(tensor, path, nrow=1, normalize=True)
