import os
import glob
import clip
import torch
import argparse
import scipy.stats
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from cleanfid import fid

def parse_args():
    desc = "Evaluation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img-folder', type=str, default='/path/to/image/folder/you/want/to/evaluate',
                        help='path to image folder that you want to evaluate.')
    parser.add_argument('--class-list', nargs='+',
                        help='type of classes that you want to evaluate', required=True, type=str)
    parser.add_argument('--device', type=int, default=0, help='gpu number')

    return parser.parse_args()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImgDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Search recursively in all subdirectories
        self.file_list = []
        for ext in ['*.png', '*.jpg']:
            self.file_list.extend(glob.glob(os.path.join(self.root_dir, '**', ext), recursive=True))

        print('Found {} generated images.'.format(len(self.file_list)))

        self.transforms = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img = self.transforms(img)
        return img

def eval(path, CLASSES, device):
    # Find all images recursively in subdirectories
    all_images = []
    for ext in ['*.png', '*.jpg']:
        all_images.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    
    print(f'Found {len(all_images)} generated images.')
    
    # Proceed even with small number of images
    if len(all_images) == 0:
        print("No images found!")
        return None
        
    eval_dataset = ImgDataset(root_dir=path)
    
    # Check dataset size
    dataset_size = len(eval_dataset)
    if dataset_size == 0:
        print("Dataset is empty!")
        return None
        
    # Set batch size with a minimum of 1
    batch_size = max(1, min(4, dataset_size))
    
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,  # Ensure minimum batch size of 1
        num_workers=2,
        drop_last=False,  # Keep all images
        pin_memory=True,
    )

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    text = clip.tokenize(CLASSES).to(device)

    img_pred_cls_list = []

    for i, data in enumerate(eval_loader):
        img = data.to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            for j in np.argmax(probs, axis=1):
                img_pred_cls_list.append(j)

    # Print results for each class
    num_each_cls_list = []
    total_images = len(img_pred_cls_list)
    if total_images > 0:  # Only proceed if we have images
        for k in range(len(CLASSES)):
            num_each_cls = len(np.where(np.array(img_pred_cls_list) == k)[0])
            num_each_cls_list.append(num_each_cls)
            ratio = num_each_cls / total_images if total_images > 0 else 0
            print(f"{CLASSES[k]}: total pred: {num_each_cls} | ratio: {ratio:.3f}")

    return num_each_cls_list

if __name__ == '__main__':
    args = parse_args()
    CLASSES_prompts = args.class_list
    device_ = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # evaluate
    num_each_cls_list = eval(args.img_folder, CLASSES_prompts, device_)
    
    if num_each_cls_list is not None and len(num_each_cls_list) > 0:
        # get the ratio
        total = np.sum(num_each_cls_list)
        if total > 0:  # Only compute metrics if we have images
            each_cls_ratio = np.array(num_each_cls_list)/total

            # compute KL
            length = len(CLASSES_prompts)
            uniform_distribution = np.ones(length)/length

            KL1 = np.sum(scipy.special.kl_div(each_cls_ratio, uniform_distribution))
            KL2 = scipy.stats.entropy(each_cls_ratio, uniform_distribution)
            
            print(f"\nFor Classes {CLASSES_prompts}, KL Divergence is {KL1:4f}")
            
            try:
                score = fid.compute_fid(args.img_folder, dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")
                print(f"FID Score is {score}")
            except Exception as e:
                print(f"Could not compute FID score: {str(e)}")
