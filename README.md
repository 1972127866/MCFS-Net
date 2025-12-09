# Sketch-Based Clothing Style Generation

- Folder: `/sketch_based_mix`
- Core Code: `embed_project.py` and `fm_mix.py`

## embed_project.py 

#### Function：

Inverse map images into the latent space of StyleGAN (convert images to feature vectors).

#### Usage Example：

```python
python embed-project.py \
    --steps=400 \
    --num_steps=250 \
    --exp experiments/encoder_cloth.json \
    --resume model_encoder/cloth_batch_8_loss_sampling_train_stylegan2ada/checkpoint/BEST_loss2.5923986434936523.pth \
    --testing_path editGANdata/toBeEmbed \
    --latent_sv_folder editGANresult/Embed_result
```

`exp` is the configuration file, where the main field to focus on is `stylegan_checkpoint`; `resume` is the trained encoder; other parameter explanations can be found in the file.

## fm_mix.py

#### Function：

Fuse the features of sketches and clothing images to generate new clothing images.

#### Usage Example：

```python
!python fm_mix.py \
    -mix_real=True \
    -source_pic_ids=10028 \
    -targets=100232 \
    -res_mix=256 \
    -styles=7-19 \
    -use_sketch=True \
    -output_path=editGANresult/fm_result_8385_samplepic \
    -sketch_path=editGANdata/latents_8385_collar \
    -latents_path editGANdata/latents_8385_collar \
    -model_path=pretrained_styleGAN/collar_cloth_8385_300t_s.pkl
```

The function of each parameter is labeled in the file.

## train_encoder.py 

#### Function：

Train the encoder. This script is generally unnecessary unless you need to train the model on a new dataset.

# Texture and Text-Based Clothing Style Generation

- Folder：/texture_and_text_based_mix
- Core Code：flexible_inference_texture_all_s_extra_mapper_img_loss.py, flexible_inference_texture_all_s_extra_mapper_img_loss_texture_for_color.py和mix_fm.py（都在scripts文件夹下）

## flexible_inference_texture_all_s_extra_mapper_img_loss.py

#### Function：

Generate clothing based on texture maps and text (where texture maps control patterns and text controls colors). If the `-target_text` parameter is not used, the texture map will simultaneously control both color and pattern.

#### Usage Example：

```python
python scripts/flexible_inference_texture_all_s_extra_mapper_img_loss.py \
    -save_name cids_1520_deep_fashion_multi_model_texture_0_100_pink_to_red_orange_yellow_green_blue \
    -cloth_ids 1520 \
    -target_text  red,orange,yellow,green,blue \
    -neutral pink \
    -texture_path /editGANdata/deep_fashion_multi_model_texture_400
```

The function of each parameter is labeled in the file. Many parameters have default values, which are set to use pre-trained StyleGAN and related models from the Fashion-Top dataset. If you want to use StyleGAN trained on the GarmentSet or other datasets, you should explicitly set the corresponding parameters.

## flexible_inference_texture_all_s_extra_mapper_img_loss_texture_for_color.py

#### Function：

Generate clothing based on texture maps and text (**where the texture map controls color and text controls patterns**).

## flexible_inference_texture_all_s_extra_mapper_img_loss_only_text.py

#### Function：

Generate clothing based on text.

## mix_fm.py

#### Function：

Fuse the features of sketches and clothing images to generate new clothing

#### Usage Example：

```python
python scripts/mix_fm.py  -source_pic_ids 5003-5006 -targets 1007,1008,1014,1006,1032,1043
```

`source_pic_ids` refers to the names of clothing image features, and `targets` refers to the names of sketch features.

---

#### Full Process for Generating Clothing Images Based on Texture Maps, Text, and Sketches:

1. Use `flexible_inference_texture_all_s_extra_mapper_img_loss.py` or `flexible_inference_texture_all_s_extra_mapper_img_loss_texture_for_color.py` to generate clothing images that meet the texture and text requirements, along with the corresponding feature vector `s`.
2. Copy the generated feature vector `s` to `/editGANdata/latent_S` and rename it as a unique filename, e.g., `5007.npy`.
3. Run `scripts/mix_fm.py`, where `-source_pic_ids` is the sequence number set in the previous step and `-targets` refers to the sketch sequence number.

---


## delta_mapper_all_s_extra_mapper.py

#### Function：

Build the Texture Delta Mapper network structure.

## get_embedding_codes_all_s.py

#### Function：

Generate training data.

#### Usage Example：

```python
CUDA_VISIBLE_DEVICES=0 python get_embedding_codes_all_s.py \
    --classname 620t_sample_200000_all_s_tmp \
    --ckpt ../editGAN/cloth-v2-620t.pt \
    --size 256 \
    --latent_dir ../editGANdata/620t_sample_200000/w \
    --use_npy True
```

`latent_dir` is a folder containing 200,000 sampled w+ space latent vectors. If you need to use a new dataset, you should create a new folder for sampling, and the sampling method can be referenced in `editGAN/sample_pics.py`.

## train_texture_all_s_extra_mapper_img_loss.py

#### Function：

Train the Texture Delta Mapper.
