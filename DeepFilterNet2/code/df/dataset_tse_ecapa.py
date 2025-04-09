  
# wujian@2018

import random
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from libdf import DF, erb, erb_norm, unit_norm

from .audio_ecapa import WaveReader
from .utils import as_real, get_norm_alpha



def make_dataloader(is_train,
                    mix_dir,
                    ref_dir,
                    batch_size: int,
                    num_workers: int,
                    chunk_size: int,
                    df_state: DF,
                    nb_df: int,
                    ):
    dataset = My_dataset(mix_dir, ref_dir, sample_rate=df_state.sr())
    return My_dataLoader(dataset,
                      is_train,
                      batch_size,
                      num_workers,
                      chunk_size,
                      df_state,
                      nb_df,
                      )


class My_dataset(Dataset):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp, ref_scp, sample_rate):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)

    
    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]

        return {
            "mix": mix.astype(np.float32),
            "ref": ref.astype(np.float32),
			"key": key
        }
    def __len__(self):
        return len(self.mix)





class My_dataLoader(object):
   
    def __init__(self,
                 dataset,
                 train,
                 batch_size,
                 num_workers,
                 chunk_size,
                 df_state,
                 nb_df,
                 ):
        self.train = train
        self.batch_size = batch_size
        self.df_state = df_state
        self.nb_df = nb_df

        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
                                    #   least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = DataLoader(dataset,
                                    batch_size=batch_size // 2,
                                    num_workers=num_workers,
                                    shuffle=train,
                                    collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            batch['noisy'], batch['feat_erb'], batch['feat_spec'] = self.df_features(batch["mix"], self.df_state, self.nb_df)
            batch['clean'], _, _ = self.df_features(batch["ref"], self.df_state, self.nb_df)
            del batch['mix'], batch['ref']
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def df_features(self, audio: torch.Tensor, df: DF, nb_df: int, mag: bool=False):

        spec = df.analysis(audio.numpy())
        if mag:
            spec = torch.as_tensor(spec)
            spec_abs = spec.abs()
            spec_abs = torch.einsum("hij->hji", spec_abs)  # (B, T, F) -> (B, F, T)
            return spec_abs
        a = get_norm_alpha(False)
        erb_fb = df.erb_widths()
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1) 
        spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
        spec = as_real(torch.as_tensor(spec).unsqueeze(1))

        return spec, erb_feat, spec_feat


    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj

class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["valid_len"] = int(N)
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks

if __name__ == "__main__":
    chunk_size = 3 * 48000
    train = True
    least = chunk_size // 2
    splitter = ChunkSplitter(chunk_size, train, least)    
    my_train_data = My_dataset('/Share/wsl/data/DC-data/path/demo/train_path/mix.scp', "/Share/wsl/data/DC-data/path/demo/train_path/ref.scp", "/Share/wsl/data/DC-data/path/demo/train_path/aux.scp", sample_rate=48000)
    # my_dev_data = My_dataset(**dev_data)
    egs = my_train_data[0]
    print(egs)

    py_df = DF(
        sr=48000, 
        fft_size=960, 
        hop_size=480, 
        nb_bands=32, 
        min_nb_erb_freqs=96,
        )
    train_loader = make_dataloader(True,
                    '/Share/wsl/data/DC-data/path/demo/train_path/mix.scp',
                    '/Share/wsl/data/DC-data/path/demo/train_path/ref.scp',
                    '/Share/wsl/data/DC-data/path/demo/train_path/aux.scp',
                    16,
                    16,
                    chunk_size,
                    py_df,
                    96,
                    )


    # chunk = splitter.split(egs)
    # dataload = My_dataLoader(my_train_data, sr=8000, chunk_size=chunk_size, batch_size= 32)
    for i, obj in enumerate(train_loader):
       
        print(obj)