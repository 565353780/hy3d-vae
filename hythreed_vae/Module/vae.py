import os

from typing import Optional


class VAE(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
        device: str='cuda:0',
    ) -> None:
        self.device = device
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][VAE::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        return True
