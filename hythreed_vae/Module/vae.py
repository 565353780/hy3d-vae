import gc
import io
import os
import sys

import numpy as np
import torch
import trimesh

from typing import Dict, Optional, Tuple, Union


# Hunyuan3D 的 `hy3dshape` 包位于 ``Hunyuan3D-2.1/hy3dshape`` 目录下（包根再嵌套一层
# ``hy3dshape``）。这里把它注入 sys.path，便于 ``import hy3dshape...``。
_HY3DSHAPE_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'Hunyuan3D-2.1', 'hy3dshape',
    )
)
if _HY3DSHAPE_ROOT not in sys.path:
    sys.path.append(_HY3DSHAPE_ROOT)

from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.models.autoencoders.model import DiagonalGaussianDistribution
from hy3dshape.pipelines import export_to_trimesh


# 官方 demo 默认采样的均匀点数（无锐边点）。
HY3D_DEFAULT_NUM_UNIFORM_POINTS = 81920
HY3D_DEFAULT_NUM_SHARP_POINTS = 0

# latents2mesh 的默认 marching cubes 参数，与 ``minimal_vae_demo.py`` 对齐。
HY3D_DEFAULT_BOUNDS = 1.01
HY3D_DEFAULT_MC_LEVEL = 0.0
HY3D_DEFAULT_NUM_CHUNKS = 20000
HY3D_DEFAULT_OCTREE_RESOLUTION = 256
HY3D_DEFAULT_MC_ALGO = 'mc'


def toSurfaceTensor(
    point_cloud: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """把任意点云规整成 Hunyuan3D encoder 期望的 ``(B, N, C)`` surface 张量。

    支持的输入形状：
    - ``(N, 3)`` / ``(B, N, 3)``：仅 xyz，自动补零法线得到 ``(B, N, 6)``。
    - ``(N, 6)`` / ``(B, N, 6)``：xyz + 法线。
    - ``(N, 7)`` / ``(B, N, 7)``：xyz + 法线 + sharp-edge label。

    与官方 ``SharpEdgeSurfaceLoader`` 输出保持同构：``[:, :, 0:3]`` 为坐标、
    ``[:, :, 3:6]`` 为法线、``[:, :, 6]``（若有）为 sharp-edge label。
    ``ShapeVAE.encode`` 内部按 ``pc=surface[:, :, :3]``、``feats=surface[:, :, 3:]``
    切分，因此 feats 维度（3 或 4）由调用方点云决定。
    """

    if isinstance(point_cloud, torch.Tensor):
        tensor = point_cloud
    else:
        tensor = torch.from_numpy(np.asarray(point_cloud))

    tensor = tensor.float()

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(
            f'toSurfaceTensor expects (N, C) or (B, N, C) point cloud, '
            f'got shape={list(tensor.shape)}.'
        )

    channels = tensor.shape[-1]
    if channels == 3:
        normals = torch.zeros_like(tensor)
        tensor = torch.cat([tensor, normals], dim=-1)
    elif channels in (6, 7):
        pass
    else:
        raise ValueError(
            f'toSurfaceTensor expects last dim in (3, 6, 7), got C={channels}.'
        )

    return tensor.contiguous()


class VAE(object):
    """Hunyuan3D 形状 VAE 封装，统一管理 encode/decode。

    输入是点云（或可采样为点云的 trimesh），输出是三角网格 ``trimesh.Trimesh``。
    封装风格与 ``flux_mv`` 的 ``SparcVAE`` 对齐：提供 Surface/PointCloud/Trimesh
    多种入口、``mean``/``std`` 数据字典、latent 张量采样以及 round-trip helper。
    """

    def __init__(
        self,
        model_file_path: Optional[str] = None,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.float16,
        num_uniform_points: int = HY3D_DEFAULT_NUM_UNIFORM_POINTS,
        num_sharp_points: int = HY3D_DEFAULT_NUM_SHARP_POINTS,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: Optional[ShapeVAE] = None

        self.surface_loader = SharpEdgeSurfaceLoader(
            num_uniform_points=num_uniform_points,
            num_sharp_points=num_sharp_points,
        )

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        """加载 Hunyuan3D ShapeVAE 权重。

        ``model_file_path`` 兼容三种形态：
        1. 指向 ``model.fp16.ckpt`` / ``*.ckpt`` / ``*.safetensors`` 的单文件，
           默认使用同目录下的 ``config.yaml``。
        2. 指向包含 ``config.yaml`` + ``model*.ckpt`` 的模型子目录。
        3. HuggingFace / 本地 repo id（如 ``tencent/Hunyuan3D-2.1``），交给
           ``ShapeVAE.from_pretrained`` 自动定位 ``hunyuan3d-vae-v2-1`` 子目录。
        """

        if os.path.isfile(model_file_path):
            use_safetensors = model_file_path.endswith('.safetensors')
            config_path = os.path.join(
                os.path.dirname(model_file_path), 'config.yaml'
            )
            if not os.path.exists(config_path):
                print('[ERROR][VAE::loadModel]')
                print('\t config.yaml not found beside ckpt!')
                print('\t expected config_path:', config_path)
                return False

            self.model = ShapeVAE.from_single_file(
                ckpt_path=model_file_path,
                config_path=config_path,
                device=str(self.device),
                dtype=self.dtype,
                use_safetensors=use_safetensors,
            )
        elif os.path.isdir(model_file_path):
            config_path = os.path.join(model_file_path, 'config.yaml')
            if not os.path.exists(config_path):
                print('[ERROR][VAE::loadModel]')
                print('\t config.yaml not found in model dir!')
                print('\t model_file_path:', model_file_path)
                return False

            ckpt_path = None
            for name in ('model.fp16.ckpt', 'model.ckpt', 'model.fp16.safetensors',
                         'model.safetensors'):
                candidate = os.path.join(model_file_path, name)
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break
            if ckpt_path is None:
                print('[ERROR][VAE::loadModel]')
                print('\t no model ckpt/safetensors found in model dir!')
                print('\t model_file_path:', model_file_path)
                return False

            self.model = ShapeVAE.from_single_file(
                ckpt_path=ckpt_path,
                config_path=config_path,
                device=str(self.device),
                dtype=self.dtype,
                use_safetensors=ckpt_path.endswith('.safetensors'),
            )
        else:
            # 当作 HuggingFace / 本地 repo id 处理。
            self.model = ShapeVAE.from_pretrained(
                model_file_path,
                device=str(self.device),
                dtype=self.dtype,
            )

        self.model.eval()
        return True

    def to(self, device) -> 'VAE':
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def cpu(self) -> 'VAE':
        return self.to('cpu')

    def cuda(self, device: Optional[str] = None) -> 'VAE':
        return self.to(device if device is not None else 'cuda')

    def _assertModelLoaded(self, caller: str) -> None:
        if self.model is None:
            raise RuntimeError(
                f'VAE.{caller} requires a loaded model; call loadModel(...) first.'
            )

    # ------------------------------------------------------------------ #
    # Encode
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encodeSurfaceData(
        self,
        surface: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """编码 surface 点云，返回 ``mean`` / ``std`` numpy 字典。

        ``surface`` 形状见 :func:`toSurfaceTensor`。这里复用 encoder + pre_kl +
        ``DiagonalGaussianDistribution`` 的底层流程拿到 ``mean`` / ``std``，
        而不是只调用一次随机采样的 ``ShapeVAE.encode``，便于后续按需重采样。
        """

        self._assertModelLoaded('encodeSurfaceData')

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        surface_tensor = toSurfaceTensor(surface).to(
            device=self.device, dtype=self.dtype
        )

        pc = surface_tensor[:, :, :3]
        feats = surface_tensor[:, :, 3:]

        latents, _ = self.model.encoder(pc, feats)
        moments = self.model.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

        mean = posterior.mean.float()
        std = posterior.std.float()

        assert not mean.isnan().any(), 'Mean contains NaN values'
        assert not std.isnan().any(), 'Std contains NaN values'

        result = dict(
            mean=mean.cpu().numpy().astype(np.float32),
            std=std.cpu().numpy().astype(np.float32),
        )

        del surface_tensor, pc, feats, latents, moments, posterior, mean, std
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def encodeSurface(
        self,
        surface: Union[np.ndarray, torch.Tensor],
        random_ratio: float = 0.0,
    ) -> torch.Tensor:
        """编码 surface 点云，返回 latent 张量（可选叠加 std 噪声）。

        ``random_ratio == 0`` 时返回分布均值，等价于 ``sample_posterior=False``；
        ``random_ratio > 0`` 时在均值上叠加 ``random_ratio * eps * std``。
        """

        sparc_data = self.encodeSurfaceData(surface=surface)
        return sampleVAELatent(sparc_data, random_ratio, device=str(self.device))

    @torch.no_grad()
    def encodePointCloudData(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """编码原始点云（xyz / xyz+normal / xyz+normal+label），返回 mean/std。"""

        return self.encodeSurfaceData(surface=point_cloud)

    @torch.no_grad()
    def encodeTrimeshData(
        self,
        mesh: trimesh.Trimesh,
    ) -> Dict[str, np.ndarray]:
        """采样 trimesh 表面为点云后编码，返回 mean/std numpy 字典。"""

        self._assertModelLoaded('encodeTrimeshData')
        surface = self.surface_loader(mesh)
        return self.encodeSurfaceData(surface=surface)

    @torch.no_grad()
    def encodeTrimeshFile(
        self,
        mesh_file_path: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not os.path.exists(mesh_file_path):
            print('[ERROR][VAE::encodeTrimeshFile]')
            print('\t mesh file not exist!')
            print('\t mesh_file_path:', mesh_file_path)
            return None

        self._assertModelLoaded('encodeTrimeshFile')
        # SharpEdgeSurfaceLoader 接受 mesh 路径，内部会 trimesh.load。
        surface = self.surface_loader(mesh_file_path)
        return self.encodeSurfaceData(surface=surface)

    @torch.no_grad()
    def encodeTrimeshStream(
        self,
        mesh_stream: io.BytesIO,
        file_type: str = 'glb',
    ) -> Optional[Dict[str, np.ndarray]]:
        mesh = trimesh.load(mesh_stream, file_type=file_type, process=False, force='mesh')
        if mesh is None or not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            print('[ERROR][VAE::encodeTrimeshStream]')
            print('\t loaded mesh from stream is invalid!')
            return None

        return self.encodeTrimeshData(mesh=mesh)

    @torch.no_grad()
    def saveVAEData(
        self,
        vae_data: Dict[str, np.ndarray],
        save_file_path: str,
    ) -> bool:
        save_dir = os.path.dirname(save_file_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        np.savez(
            save_file_path,
            mean=vae_data['mean'].astype(np.float32),
            std=vae_data['std'].astype(np.float32),
        )
        return True

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def decodeVAELatent(
        self,
        vae_latent: torch.Tensor,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        """把 VAE latent 解码为三角网格 ``trimesh.Trimesh``。

        ``vae_latent`` 形状 ``(num_latents, embed_dim)`` 或带 batch 的
        ``(B, num_latents, embed_dim)``（仅取第 0 个样本输出）。流程与官方
        ``minimal_vae_demo`` 对齐：``decode`` -> ``latents2mesh``（体素解码 +
        Marching Cubes）-> ``export_to_trimesh``。
        """

        self._assertModelLoaded('decodeVAELatent')

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        vae_latent = vae_latent.to(device=self.device, dtype=self.dtype)
        if vae_latent.dim() == 2:
            vae_latent = vae_latent.unsqueeze(0)
        if vae_latent.dim() != 3:
            raise ValueError(
                f'VAE.decodeVAELatent expects (num_latents, C) or '
                f'(B, num_latents, C) latent, got shape={list(vae_latent.shape)}.'
            )

        latents = self.model.decode(vae_latent)
        mesh_output = self.model.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

        meshes = export_to_trimesh(mesh_output)
        if isinstance(meshes, list):
            tri_mesh = meshes[0] if len(meshes) > 0 else None
        else:
            tri_mesh = meshes

        del vae_latent, latents, mesh_output, meshes
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return tri_mesh

    @torch.no_grad()
    def decodeVAEData(
        self,
        vae_data: Dict[str, np.ndarray],
        random_ratio: float = 0.0,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        vae_latent = sampleVAELatent(vae_data, random_ratio, device=str(self.device))
        return self.decodeVAELatent(
            vae_latent=vae_latent,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

    @torch.no_grad()
    def decodeVAELatentStream(
        self,
        vae_latent_stream: io.BytesIO,
        random_ratio: float = 0.0,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        vae_data = np.load(vae_latent_stream)
        return self.decodeVAEData(
            vae_data=vae_data,
            random_ratio=random_ratio,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

    @torch.no_grad()
    def decodeVAELatentFile(
        self,
        vae_latent_file_path: str,
        random_ratio: float = 0.0,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        if not os.path.exists(vae_latent_file_path):
            print('[ERROR][VAE::decodeVAELatentFile]')
            print('\t vae latent file not exist!')
            print('\t vae_latent_file_path:', vae_latent_file_path)
            return None

        vae_data = np.load(vae_latent_file_path)
        return self.decodeVAEData(
            vae_data=vae_data,
            random_ratio=random_ratio,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )

    # ------------------------------------------------------------------ #
    # Round-trip helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def reconstructTrimesh(
        self,
        mesh: trimesh.Trimesh,
        random_ratio: float = 0.0,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[Tuple[Dict[str, np.ndarray], trimesh.Trimesh]]:
        """Encode -> Decode 一站式接口，便于做 VAE round-trip。"""

        vae_data = self.encodeTrimeshData(mesh=mesh)
        if vae_data is None:
            return None
        vae_mesh = self.decodeVAEData(
            vae_data=vae_data,
            random_ratio=random_ratio,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )
        return vae_data, vae_mesh

    @torch.no_grad()
    def reconstructPointCloud(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
        random_ratio: float = 0.0,
        bounds: float = HY3D_DEFAULT_BOUNDS,
        mc_level: float = HY3D_DEFAULT_MC_LEVEL,
        num_chunks: int = HY3D_DEFAULT_NUM_CHUNKS,
        octree_resolution: int = HY3D_DEFAULT_OCTREE_RESOLUTION,
        mc_algo: str = HY3D_DEFAULT_MC_ALGO,
        enable_pbar: bool = True,
    ) -> Optional[Tuple[Dict[str, np.ndarray], trimesh.Trimesh]]:
        vae_data = self.encodePointCloudData(point_cloud=point_cloud)
        if vae_data is None:
            return None
        vae_mesh = self.decodeVAEData(
            vae_data=vae_data,
            random_ratio=random_ratio,
            bounds=bounds,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=enable_pbar,
        )
        return vae_data, vae_mesh


def sampleVAELatent(
    vae_data: Dict,
    random_ratio: float = 0.0,
    device: str = 'cuda:0',
) -> torch.Tensor:
    """从 ``mean`` / ``std`` 字典采样出 latent 张量。

    ``random_ratio == 0`` 时返回均值（确定性）；``random_ratio > 0`` 时叠加
    ``random_ratio * eps * std`` 噪声。
    """

    mean = vae_data['mean']
    std = vae_data['std']

    if isinstance(mean, np.ndarray):
        feat = torch.from_numpy(mean).to(device)
    else:
        feat = mean.to(device)
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).to(device)
    else:
        std = std.to(device)

    feat = feat.clone()
    if random_ratio > 0:
        sample = torch.randn(feat.shape, device=feat.device, dtype=feat.dtype)
        feat = feat + random_ratio * sample * std

    return feat.float()
