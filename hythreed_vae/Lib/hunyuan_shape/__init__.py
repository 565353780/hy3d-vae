# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# 这是从上游 Hunyuan3D-2.1/hy3dshape 抽取的最小 VAE 推理闭包，仅保留
# mesh/点云 -> ShapeVAE encode -> latent -> decode -> trimesh 所需的代码。
# 故意保持包入口为空，避免触发上游 __init__ 对 DiT pipeline、diffusers、
# transformers、postprocessors 等的副作用导入。
