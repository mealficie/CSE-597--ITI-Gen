from pathlib import Path
from shutil import copy2

ORIGINAL_DATASET_PATH = Path("data/celeba/Eyeglasses")
NEG_POS_FILENAMES_PATH = Path("generate_jobfiles/diversity_eyeglasses/Eyeglasses_negative_Male_positive.txt")
POS_POS_FILENAMES_PATH = Path("generate_jobfiles/diversity_eyeglasses/Eyeglasses_positive_Male_positive.txt")

neg_filenames = [file.name for file in (ORIGINAL_DATASET_PATH / "negative").iterdir() if file.is_file()]
pos_filenames = [file.name for file in (ORIGINAL_DATASET_PATH / "positive").iterdir() if file.is_file()]

with open(NEG_POS_FILENAMES_PATH, "r") as f:
    pos_neg_filenames = f.read().splitlines()
with open(POS_POS_FILENAMES_PATH, "r") as f:
    pos_pos_filenames = f.read().splitlines()
neg_neg_filenames = [filename for filename in neg_filenames if filename not in pos_neg_filenames and ".jpg" in str(filename)]

non_diverse_path = Path(str(ORIGINAL_DATASET_PATH).rpartition("/")[0] + "/Eyeglasses_non_diverse")
only_male_path = Path(str(ORIGINAL_DATASET_PATH).rpartition("/")[0] + "/Eyeglasses_Male_only_positive")

(non_diverse_path / "negative").mkdir(parents=True, exist_ok=True)
(non_diverse_path / "positive").mkdir(parents=True, exist_ok=True)
(only_male_path / "negative").mkdir(parents=True, exist_ok=True)
(only_male_path / "positive").mkdir(parents=True, exist_ok=True)

for filename in neg_neg_filenames:
    target_path = non_diverse_path / "negative" / filename
    if target_path.exists():
        target_path.unlink()
    copy2(ORIGINAL_DATASET_PATH / "negative" / filename, target_path)

for filename in pos_neg_filenames:
    target_path = only_male_path / "negative" / filename
    if target_path.exists():
        target_path.unlink()
    copy2(ORIGINAL_DATASET_PATH / "negative" / filename, target_path)

for filename in pos_pos_filenames:
    non_diverse_target = non_diverse_path / "positive" / filename
    only_male_target = only_male_path / "positive" / filename
    
    if non_diverse_target.exists():
        non_diverse_target.unlink()
    if only_male_target.exists():
        only_male_target.unlink()
        
    copy2(ORIGINAL_DATASET_PATH / "positive" / filename, non_diverse_target)
    copy2(ORIGINAL_DATASET_PATH / "positive" / filename, only_male_target)
