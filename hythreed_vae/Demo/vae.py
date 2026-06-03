import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from hythreed_vae.Module.vae import VAE


def demo():
    home = '/mnt/EFS2/share_89f07eb4_6378_4d3c_8c02/lichanghao/'
    model_file_path = home + 'chLi/Model/HY3D/vae/model.fp16.ckpt'
    device = 'cuda:0'

    vae = VAE(
        model_file_path=model_file_path,
        device=device,
    )
    return True
