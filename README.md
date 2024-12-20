# Replicating ITI-Gen for my CSE-597 Final Project

## Installation
The code has been tested with the following environment:
```angular2html
git clone repo-link
cd repo
conda env create --name env-name --file=environment.yml
source activate env-name
```


## Data Preparation

1. The following datasets are used in replication of the project ITI-Gen:

|   Dataset    |      Description      |       Attribute Used        |                                        Google Drive                                        |
|:------------:|:---------------------:|:---------------------------:|:------------------------------------------------------------------------------------------:|
|  [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |   Real face images    | 40 binary facial attributes | [Link](https://drive.google.com/file/d/1_wxcrzirofEge4i8LTyYBAL0SMQ_LwGO/view?usp=sharing) | 
| [FairFace](https://github.com/joojs/fairface) |   Real face images    |    Age with 9 categories    | [Link](https://drive.google.com/file/d/1_xtui0b0O52u38jbJzrxW8yRRiBHnZaA/view?usp=sharing) |
|   [FAIR](https://trust.is.tue.mpg.de/)   | Synthetic face images |   Skin tone with 6 categories    | [Link](https://drive.google.com/file/d/1_wiqq7FDByLp8Z4WQOeboSEXYsCzmV76/view?usp=sharing) |

The reference images are stored under ```data/``` directory in this order:
```angular2html
|-- data
|   |-- celeba
|   |   |-- 5_o_Clock_Shadow
|   |   |-- Bald
|   |   |-- ...

|   |-- FAIR_benchmark
|   |   |-- Skin_tone

|   |-- fairface
|   |   |-- Age
```

2. **(Optional)** You can also construct _customized_ reference images under ```data/``` directory:
```angular2html
|-- data
|   |-- custom_dataset_name
|   |   |-- Attribute_1
|   |   |   |-- Category_1
|   |   |   |-- Category_2
|   |   |   |-- ..
|   |   |-- Attribute_2
|   |   |-- ...
```
Modify the corresponding functions in `util.py`.




## Training ITI-GEN



**Train on human domain**
```shell
python train_iti_gen.py \
    --prompt='a headshot of a person' \
    --attr-list='Male,Age' \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
  - `--prompt`: prompt that you want to debias.
  - `--attr_list`: attributes should be selected from `Dataset_name_attribute_list` in `util.py`, separated by commas. Empirically, attributes that are easier to train (less # of category, easier to tell the visual difference between categories) should be put in the front, eg. Male < Young < ... < Skin_Tone < Age.

  - Pre-trained checkpoints used in the experiment for single-attribute Young in CelebA and multi-attribute Male,Young and the combination Male x Eyeglasses when using different reference datasets for the Eyeglasses attribute are availble in ```chkpts/`` folder



## Generation
ITI-GEN training is standalone from the generative models such as Stable Diffusion, ControlNet, and InstructionPix2Pix.
Here I have used ITI-GEN to generate images with Stable Diffusion, Stable Diffusion model has to be added to ```models/``` directory to generate the images. To add any other model you can follow the step below like i did for 
Stable Diffusion
### Stable Diffusion installation
```shell
cd models
git clone https://github.com/CompVis/stable-diffusion.git
# ITI-GEN has been tested with this version: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
# Due to licence issues, we cannot share the pre-trained checkpoints directly.
mv stable-diffusion sd
mkdir -p sd/models/ldm/stable-diffusion-v1/
ln -s <path/to/sd-v1-4.ckpt> sd/models/ldm/stable-diffusion-v1/model.ckpt
cd sd
pip install -e .
cd ../..
```

### Image generation

**Generation on the human domain**

```shell
python generation.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --attr-list='Male,Skin_tone,Age' \
    --outdir='./ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/original_prompt_embedding/sample_results' \
    --prompt-path='./ckpts/a_headshot_of_a_person_Male_Skin_tone_Age/original_prompt_embedding/basis_final_embed_19.pt' \
    --n_iter=5 \
    --n_rows=5 \
    --n_samples=1
```
- `--config`: config file for Stable Diffusion.
- `--ckpt`: path to the pre-trained Stable Diffusion checkpoint.
- `--plms`: whether to use the plms sampling.
- `--attr_list`: attributes should be selected from `Dataset_name_attribute_list` in `util.py`, separated by commas. This should align with the attribute list used in training ITI-GEN.
- `--outdir`: output directory of the generated images.
- `--prompt_path`: path to the learnt prompt embeddings with ITI-GEN.
- `--n_iter`: number of iterations for the diffusion sampling.
- `--n_rows`: number of rows in the output image grid.
- `--n_samples`: number of samples per row.

- The images generated in the experiment for single-attribute Young in CelebA and multi-attribute Male,Young and the combination Male x Eyeglasses when using different reference datasets for the Eyeglasses attribute are availble in ```results/`` folder


## Evaluation
We show using CLIP, is used for evaluating most of the attributes. 
When it might be erroneous for some attributes, we combine the CLIP results with human evaluations.
The output for this script contains the quantitative results of both `KL divergence` and `FID` score, supported by [CleanFID](https://github.com/GaParmar/clean-fid).

```shell
python evaluation.py \
    --img-folder '/path/to/image/folder/you/want/to/evaluate' \
    --class-list 'a headshot of a person wearing eyeglasses' 'a headshot of a person'
```
- `--img_folder`: the image folder that you want to evaluate.
- `--class_list`: the class prompts used for evaluation, separated by a space. The length of the list depends on the number of category combinations for different attributes. In terms of writing evaluation prompts for CelebA attributes, please refer (but not limited) to Table A3 in the supplementary materials.

We should notice FID score can be affected by various factors such as the image number. 

## Acknowledgements
- Models
  - [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- Data acquisition and processing
  - [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [FairFace](https://github.com/joojs/fairface)
  - [FAIR](https://trust.is.tue.mpg.de/)
  - [CLIP-IQA](https://github.com/IceClear/CLIP-IQA)

## Results
you can find the results reported in my report from the two python notebook in the repo.

## License
MIT License

Copyright (c) [2024] [mealficie]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
