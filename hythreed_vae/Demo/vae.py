import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import trimesh

from hythreed_vae.Module.vae import VAE


def demo():
    home = '/mnt/EFS2/share_89f07eb4_6378_4d3c_8c02/lichanghao/'
    # 模型文件夹内需同时包含 model.fp16.ckpt 和 config.yaml。
    model_folder_path = home + 'chLi/Model/HY3D/vae/'
    device = 'cuda:0'

    # 输入是点云（这里用一个 mesh 采样得到点云做演示），输出是三角网格。
    mesh_file_path = '../Hunyuan3D-2.1/hy3dshape/demos/demo.glb'
    save_folder_path = home + 'chLi/Results/HY3D/vae/test/'
    os.makedirs(save_folder_path, exist_ok=True)

    random_ratio = 0.0
    octree_resolution = 256

    vae = VAE(
        model_folder_path=model_folder_path,
        device=device,
    )

    # 路径一：直接从 mesh 文件采样点云并 encode。
    vae_data = vae.encodeTrimeshFile(mesh_file_path=mesh_file_path)
    if vae_data is None:
        print('[ERROR][demo::vae] encode failed!')
        return False

    vae.saveVAEData(
        vae_data=vae_data,
        save_file_path=save_folder_path + 'vae_latent.npz',
    )

    vae_mesh = vae.decodeVAEData(
        vae_data=vae_data,
        random_ratio=random_ratio,
        octree_resolution=octree_resolution,
    )
    if vae_mesh is None:
        print('[ERROR][demo::vae] decode failed!')
        return False

    vae_mesh.export(save_folder_path + 'vae_mesh.obj')

    # 路径二：一站式 round-trip helper（输入已经加载好的 trimesh）。
    gt_mesh = trimesh.load(mesh_file_path, process=False, force='mesh')
    result = vae.reconstructTrimesh(
        mesh=gt_mesh,
        random_ratio=random_ratio,
        octree_resolution=octree_resolution,
    )
    if result is not None:
        _, recon_mesh = result
        recon_mesh.export(save_folder_path + 'vae_mesh_roundtrip.obj')

    print('[INFO][demo::vae] done, results saved to:', save_folder_path)
    return True
