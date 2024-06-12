import os
from typing import Union, Tuple
import pickle

import torch
from safetensors.torch import save_file, safe_open

import bz2file as bz2

class Recorder:
    EMPTY_HANDLER_SKIP: str = "SKIP"
    EMPTY_HANDLER_ERR: str = "ERR"
    EMPTY_HANDLER_DEFAULT: str = "DEFAULT"
    def __init__(self) -> None:
        self.__data__: dict = {}
        
    def __init_data_by_key__(self, key: Union[str, int]) -> None:
        if not key in self.__data__:
            self.__data__[key] = {}
    
    def __handle_values__(self, values, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice]):
        if isinstance(indices, int) or isinstance(indices, float):
            return [values]
        return values
    
    def __handle_indices__(self, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None):
        if indice_max_length is None:
            indice_max_length = len(self.__data__)
        if isinstance(indices, slice):
            indices = [x for x in range(*indices.indices(indice_max_length))]
        elif not hasattr(indices, '__len__'):
            indices = [indices]
        return indices
    
    def __handle_values_indices__(self, values, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None):
        return self.__handle_values__(values=values, indices=indices), self.__handle_indices__(indices=indices, indice_max_length=indice_max_length)
    
    def __update_by_indices__(self, key: Union[str, int], values, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], err_if_replace: bool=False, replace: bool=False):
        if err_if_replace and replace:
            raise ValueError(f"Arguement err_if_replace and replace shouldn't be true at the same time.")
        
        for i, idx in enumerate(indices):
            if idx in self.__data__[key]:
                if replace and not err_if_replace:
                    self.__data__[key][idx] = values[i]
                elif not replace and err_if_replace:
                    raise ValueError(f"Cannot update existing value with key: {key} and indice: {idx}")
            else:
                self.__data__[key][idx] = values[i]
    
    def update_by_key(self, key: Union[str, int], values, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, err_if_replace: bool=False, replace: bool=False) -> None:
        if err_if_replace and replace:
            raise ValueError(f"Arguement err_if_replace and replace shouldn't be true at the same time.")
        self.__init_data_by_key__(key=key)
        
        values, indices = self.__handle_values_indices__(values=values, indices=indices, indice_max_length=indice_max_length)
        # print(f"Handled values: {values}, Handled indices: {indices}")
            
        if len(indices) != len(values):
            raise ValueError(f"values and indices should have the same length.")
        self.__update_by_indices__(key=key, values=values, indices=indices, err_if_replace=err_if_replace, replace=replace)
        
    def get_by_key(self, key: Union[str, int], indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, empty_handler: str=EMPTY_HANDLER_DEFAULT, default_val=None):
        self.__init_data_by_key__(key=key)
        
        indices = self.__handle_indices__(indices=indices, indice_max_length=indice_max_length)
        ret_ls = []
        for idx in indices:
            if idx in self.__data__[key]:
                # print(f"Get [{key}][{idx}]: {self.__data__[key][idx]}")
                ret_ls.append(self.__data__[key][idx])
            elif empty_handler == Recorder.EMPTY_HANDLER_DEFAULT:
                # print(f"Get [{key}][{idx}]: None")
                ret_ls.append(default_val)
            elif empty_handler == Recorder.EMPTY_HANDLER_ERR:
                raise ValueError(f"Value at [{key}][{idx}] is empty.")
        return ret_ls
    
class TensorDict:
    __EMBED_KEY__ = "EMBED"
    __DATA_KEY__ = "DATA"
    def __init__(self, max_size: int=10000) -> None:
        self.__max_size__: int = max_size
        
        # self.__embed__: torch.nn.Embedding = torch.nn.Embedding(num_embeddings=self.__max_size__, embedding_dim=1)
        self.__data__: dict[list, any] = {}
        
    def __get_key__(self, key: torch.Tensor) -> int:
        # print(f"key: {hash(tuple(key.reshape(-1).tolist()))}")
        return hash(tuple(key.reshape(-1).tolist()))
    
    def __getitem__(self, key: torch.Tensor):
        return self.__data__[self.__get_key__(key)]
    
    def __setitem__(self, key: torch.Tensor, value: any):
        self.__data__[self.__get_key__(key)] = value
    
    def is_key_exist(self, key: torch.Tensor) -> bool:
        embed_key = self.__get_key__(key)
        return embed_key in self.__data__
    
    def __pack_internal__(self) -> dict:
        # return {TensorDict.__EMBED_KEY__: self.__embed__, TensorDict.__DATA_KEY__: self.__data__}
        return {TensorDict.__DATA_KEY__: self.__data__}
    
    def __unpack_internal__(self, input: dict) -> None:
        # self.__embed__ = input[TensorDict.__EMBED_KEY__]
        self.__data__ = input[TensorDict.__DATA_KEY__]
    
    def save(self, path: Union[str, os.PathLike], file_ext: str='pkl') -> None:
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        pickle.dump(self.__pack_internal__(), file_path, pickle.HIGHEST_PROTOCOL)
        
    def load(self, path: Union[str, os.PathLike]) -> None:
        with open(path, 'rb') as f:
            self.__unpack_internal__(input=pickle.load(f))
            
class DirectRecorder:
    TOP_DICT_KEY: str = "TOP_DICT"
    TOP_DICT_MAX_SIZE_KEY: str = "TOP_DICT_MAX_SIZE"
    SUB_DICT_MAX_SIZE_KEY: str = "SUB_DICT_MAX_SIZE"
    
    SEQ_KEY: str = 'SEQ'
    RECONST_KEY: str = 'RECONST'
    IMAGE_KEY: str = 'IMAGE'
    NOISE_KEY: str = 'NOISE'
    NOISY_IMAGE_KEY: str = 'NOISY_IMAGE'
    TS_KEY: str = "ts"
    LABEL_KEY: str = "LABEL"
    def __init__(self, top_dict_max_size: int=10000, sub_dict_max_size: int=10000) -> None:
        self.__top_dict__: TensorDict[torch.Tensor, TensorDict[torch.Tensor, dict]] = TensorDict(max_size=top_dict_max_size)
        self.__top_dict_max_size__: int = top_dict_max_size
        self.__sub_dict_max_size__: int = sub_dict_max_size
        
    def __getitem__(self, key: torch.Tensor) -> TensorDict:
        if self.__top_dict__.is_key_exist(key=key):
            return self.__top_dict__[key]
        else:
            self.__top_dict__[key] = TensorDict(max_size=self.__sub_dict_max_size__)
            return self.__top_dict__[key]
    
    def __init_key__(self, top_key: torch.Tensor, sub_key: torch.Tensor):
        if top_key is None or sub_key is None:
            raise TypeError("")
        if not self.__top_dict__.is_key_exist(key=top_key):
            self.__top_dict__[top_key] = TensorDict(max_size=self.__sub_dict_max_size__)
            
        if not self.__top_dict__[top_key].is_key_exist(key=sub_key):
            self.__top_dict__[top_key][sub_key] = {DirectRecorder.SEQ_KEY: Recorder(), DirectRecorder.RECONST_KEY: None}
    
    def update_seq(self, top_key: torch.Tensor, sub_key: torch.Tensor, values, indices: Union[int, float, list[Union[int, float]], torch.Tensor, slice], indice_max_length: int=None, err_if_replace: bool=False, replace: bool=False):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.SEQ_KEY].update_by_key(key='seq', values=values, indices=indices, indice_max_length=indice_max_length, err_if_replace=err_if_replace, replace=replace)        
    
    def update_seq(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.SEQ_KEY].update_by_key(key='seq', values=values, indices=[i for i in range(len(values))], indice_max_length=None, err_if_replace=False, replace=False)
        
    def update_reconst(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.RECONST_KEY] = values
        
    def update_noise(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.NOISE_KEY] = values
        
    def update_ts(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.TS_KEY] = values
        
    def update_label(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.LABEL_KEY] = values
        
    def update_image(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.IMAGE_KEY] = values
        
    def update_noisy_image(self, top_key: torch.Tensor, sub_key: torch.Tensor, values):
        self.__init_key__(top_key, sub_key)
        self.__top_dict__[top_key][sub_key][DirectRecorder.NOISY_IMAGE_KEY] = values
        
    def batch_update(self, top_keys: torch.Tensor, sub_keys: torch.Tensor, seq: torch.Tensor, reconst: torch.Tensor, noise: torch.Tensor, ts: int, label: str):
        Ts: int = seq.shape[1]
        for i, (top_key, sub_key) in enumerate(zip(top_keys, sub_keys)):
            # print(f"top_key: {top_key.shape}, sub_key: {sub_key.shape}")
            # self.set_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[:, i, :, :, :]), indices=[i for i in range(Ts)])
            self.update_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[i]))
            self.update_reconst(top_key=top_key, sub_key=sub_key, values=reconst)
            self.update_noise(top_key=top_key, sub_key=sub_key, values=noise)
            self.update_image(top_key=top_key, sub_key=sub_key, values=top_key)
            self.update_noisy_image(top_key=top_key, sub_key=sub_key, values=sub_key)
            self.update_ts(top_key=top_key, sub_key=sub_key, values=ts)
            self.update_label(top_key=top_key, sub_key=sub_key, values=label)
            
    def __pack_internal__(self) -> dict:
        return {DirectRecorder.TOP_DICT_KEY: self.__top_dict__, DirectRecorder.TOP_DICT_MAX_SIZE_KEY: self.__top_dict_max_size__, DirectRecorder.SUB_DICT_MAX_SIZE_KEY: self.__sub_dict_max_size__}
    
    def __unpack_internal__(self, input: dict) -> None:
        self.__top_dict__ = input[DirectRecorder.TOP_DICT_KEY] 
        self.__top_dict_max_size__ = input[DirectRecorder.TOP_DICT_MAX_SIZE_KEY]
        self.__sub_dict_max_size__ = input[DirectRecorder.SUB_DICT_MAX_SIZE_KEY] 
    
    def save(self, path: Union[str, os.PathLike], file_ext: str='pkl') -> None:
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        # with open(file_path, "wb") as f:
        with bz2.BZ2File(file_path, 'w') as f:
            pickle.dump(self.__pack_internal__(), f, pickle.HIGHEST_PROTOCOL)
        # save_file(self.__pack_internal__(), file_path)
        
    def load(self, path: Union[str, os.PathLike]) -> None:
        # with open(path, 'rb') as f:
        with bz2.BZ2File(path, 'rb') as f:
            self.__unpack_internal__(input=pickle.load(f))
        # loaded_data: dict = {}
        # with safe_open(path, framework="pt", device='cpu') as f:
        #     for k in f.keys():
        #         loaded_data[k] = f.get_tensor(k)
        #     self.__unpack_internal__(input=loaded_data)
        
class SafetensorRecorder(DirectRecorder):
    PROC_BEF_SAVE_MODE_STACK: str = "stack"
    PROC_BEF_SAVE_MODE_CAT: str = "cat"
    def __init__(self) -> None:
        self.__data__ = {}
        
    def __pack_internal__(self) -> dict:
        # self.process_before_saving(mode=proc_mode)
        return self.__data__
    
    def __unpack_internal__(self, input: dict) -> None:
        self.__data__: dict[str, torch.Tensor] = input
        
    def update_seq(self, values: torch.Tensor):
        if SafetensorRecorder.SEQ_KEY in self.__data__:
            self.__data__[SafetensorRecorder.SEQ_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.SEQ_KEY] = [values]
        
    def update_reconst(self, values: torch.Tensor):
        if SafetensorRecorder.RECONST_KEY in self.__data__:
            self.__data__[SafetensorRecorder.RECONST_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.RECONST_KEY] = [values]
        
    def update_noise(self, values: torch.Tensor):
        if SafetensorRecorder.NOISE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.NOISE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.NOISE_KEY] = [values]
        
    def update_ts(self, values: torch.Tensor):
        if SafetensorRecorder.TS_KEY in self.__data__:
            self.__data__[SafetensorRecorder.TS_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.TS_KEY] = [values]
        
    def update_label(self, values: torch.Tensor):
        if SafetensorRecorder.LABEL_KEY in self.__data__:
            self.__data__[SafetensorRecorder.LABEL_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.LABEL_KEY] = [values]
        
    def update_image(self, values: torch.Tensor):
        if SafetensorRecorder.IMAGE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.IMAGE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.IMAGE_KEY] = [values]
        
    def update_noisy_image(self, values: torch.Tensor):
        if SafetensorRecorder.NOISY_IMAGE_KEY in self.__data__:
            self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY].append(values)
        else:
            self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = [values]
        
    def batch_update(self, images: torch.Tensor, noisy_images: torch.Tensor, seqs: torch.Tensor, reconsts: torch.Tensor, noises: torch.Tensor, timestep: int, label: str):
        n: int = len(images)
        labels: list[str] = [label] * n
        tss: list[int] = [timestep] * n
        for i, (image, noisy_image, noise, reconst, seq, lab, ts) in enumerate(zip(images, noisy_images, noises, reconsts, seqs, labels, tss)):
            # print(f"top_key: {top_key.shape}, sub_key: {sub_key.shape}")
            # self.set_seq(top_key=top_key, sub_key=sub_key, values=torch.squeeze(seq[:, i, :, :, :]), indices=[i for i in range(Ts)])
            # print(f"Update: ts: {torch.LongTensor([ts])}, label: {torch.LongTensor([lab])}")
            self.update_seq(values=seq)
            self.update_reconst(values=reconst)
            self.update_noise(values=noise)
            self.update_image(values=image)
            self.update_noisy_image(values=noisy_image)
            self.update_ts(values=torch.LongTensor([ts]))
            self.update_label(values=torch.LongTensor([lab]))
            
    def process_before_saving(self, mode: str):
        if mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK:
            for key, val in self.__data__.items():
                self.__data__[key] = torch.stack(val, dim=0)
        elif mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT:
            for key, val in self.__data__.items():
                self.__data__[key] = torch.cat(val, dim=0)
        else:
            raise ValueError(f"Arguement mode should be {SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK} or {SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT}")
            
    def save(self, path: Union[str, os.PathLike], file_ext: str='safetensors', proc_mode: str=PROC_BEF_SAVE_MODE_CAT) -> None:
        file_path: str = f"{path}.{file_ext}"
        if file_ext is None or file_ext == "":
            file_path: str = path
        # with open(file_path, "wb") as f:
        # with bz2.BZ2File(file_path, 'w') as f:
        #     pickle.dump(self.__pack_internal__(), f, pickle.HIGHEST_PROTOCOL)
        self.process_before_saving(mode=proc_mode)
        save_file(self.__pack_internal__(), file_path)
        
    def load(self, path: Union[str, os.PathLike], enable_update: bool=False) -> 'SafetensorRecorder':
        # with open(path, 'rb') as f:
        # with bz2.BZ2File(path, 'rb') as f:
        #     self.__unpack_internal__(input=pickle.load(f))
        loaded_data: dict = {}
        with safe_open(path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if enable_update:
                    loaded_data[k] = [f.get_tensor(k)]
                else:
                    loaded_data[k] = f.get_tensor(k)
            self.__unpack_internal__(input=loaded_data)
        return self

if __name__ == "__main__":
    rec: Recorder = Recorder()
    rec.update_by_key(key='seq', values=[1], indices=[1])
    assert rec.get_by_key(key='seq', indices=[0, 1]) == [None, 1]
    rec.update_by_key(key='seq', values=[3], indices=[3])
    assert rec.get_by_key(key='seq', indices=[0, 1, 2, 3]) == [None, 1, None, 3]
    assert rec.get_by_key(key='seq', indices=[0, 1, 3]) == [None, 1, 3]
    assert rec.get_by_key(key='seq', indices=[0, 1, 2]) == [None, 1, None]
    assert rec.get_by_key(key='seq', indices=[0, 1, 1]) == [None, 1, 1]
    assert rec.get_by_key(key='seq', indices=[0, 1, 0]) == [None, 1, None]
    
    # print(f"rec :{}")
        