import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def randAugment(N, M, p, cut_out = False, normalize=False, tensor=False):
    # Magnitude(M) search space  
    
    sola = np.linspace(0,256,10)
    post = [4,4,5,5,6,6,7,7,8,8]
    cont = [np.linspace(-0.8,-0.1,10),np.linspace(0.1,2,10)]
    bright = np.linspace(0.1,0.7,10)
    shar = np.linspace(0.1,0.9,10)
    cut = np.linspace(0,20,10)
    # Transformation search space
    Aug =[
        A.Equalize(p=p),
        A.Solarize(threshold=sola[M], p=p),
        A.Posterize(num_bits=post[M], p=p),
        A.RandomBrightnessContrast(contrast_limit=[cont[0][M], cont[1][M]], brightness_limit=bright[M], p=p),
        A.Sharpen(alpha=shar[M], lightness=shar[M], p=p)]
    # Sampling from the Transformation search space
    ops = np.random.choice(Aug, N)
    ops = list(ops)
    if cut_out:
        ops.append(A.CoarseDropout(max_holes=M, max_height=int(cut[M]), max_width=int(cut[M]), p=p))
    if normalize:
        ops.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p = 1.0))
    if tensor:
        ops.append(ToTensorV2())
    transforms = A.Compose(ops)
    return transforms, ops