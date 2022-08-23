import pytorch_lightning as pl
import pandas as pd
from os import listdir
from os.path import splitext, join


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_folder: str, val_folder: str, train_metadata_path: str = None, val_metadata_path: str = None,
                 filter_user: int = None, only_fingers: bool = False):
        super(BaseDataModule, self).__init__()
        self.train_samples = self.__filter_samples(train_folder, train_metadata_path, filter_user, only_fingers)
        self.val_samples = self.__filter_samples(val_folder, val_metadata_path, filter_user, only_fingers)

    @staticmethod
    def __filter_samples(data_folder, metadata_path, filter_user=None, only_fingers=False):
        samples = {splitext(sample)[0] for sample in listdir(data_folder)}
        if metadata_path is None:
            return [join(data_folder, f'{sample}.npz') for sample in samples]

        metadata = pd.read_csv(metadata_path)
        if filter_user is not None:
            user_samples = set(metadata['fname'][metadata.userid == filter_user])
            samples &= user_samples
        if only_fingers:
            fingers_samples = set(metadata['fname'][metadata.fingers == 'finger_incl'])
            samples &= fingers_samples
        samples = [join(data_folder, f'{train_file}.npz') for train_file in samples]
        return samples
