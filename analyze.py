# %%
import os
from typing import Union, Tuple
import pathlib
import argparse

import pandas as pd
import torch
import numpy as np

from tqdm import tqdm

from detect import plot_scatter, plot_run
from util import DirectRecorder, TensorDict, SafetensorRecorder, mse, mse_series, center_mse_series

def show_col_shapes(df: pd.DataFrame):
    for key, val in recorder.__data__.items():
        print(f"[{key}]: {val.shape}")
        
def reshape_recorder(recorder: SafetensorRecorder):
    recorder.__data__[SafetensorRecorder.IMAGE_KEY] = recorder.__data__[SafetensorRecorder.IMAGE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.NOISE_KEY] = recorder.__data__[SafetensorRecorder.NOISE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = recorder.__data__[SafetensorRecorder.NOISY_IMAGE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.RECONST_KEY] = recorder.__data__[SafetensorRecorder.RECONST_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.LABEL_KEY] = recorder.__data__[SafetensorRecorder.LABEL_KEY].reshape(-1)
    recorder.__data__[SafetensorRecorder.SEQ_KEY] = recorder.__data__[SafetensorRecorder.SEQ_KEY].reshape(-1, ts, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.TS_KEY] = recorder.__data__[SafetensorRecorder.TS_KEY].reshape(-1)
    return recorder

def load_cache_df(file: str, err_if_not_exist: bool=False):
    is_file_exist: bool = os.path.isfile(file)
    if err_if_not_exist and (not is_file_exist):
        raise ValueError(f"File, {file} does not exist.")
    elif is_file_exist:
        return pd.read_hdf(file, key='df')
    else:
        return None

def save_cache_df(df: pd.DataFrame, file: str, overwrite: bool=True):
    if overwrite or (not os.path.isfile(file)):
        df.to_hdf(file, key='df')
        
def x_y_set_list_gen(x_label: str, y_label: str, real_n: int, out_dist_n: int, fake_n: int):
    x_set_list = [torch.tensor(df[x_label][:real_n].values), torch.tensor(df[x_label][real_n:real_n + out_dist_n].values), torch.tensor(df[x_label][real_n + out_dist_n:].values)]
    y_set_list = [torch.tensor(df[y_label][:real_n].values), torch.tensor(df[y_label][real_n:real_n + out_dist_n].values), torch.tensor(df[y_label][real_n + out_dist_n:].values)]
    return x_set_list, y_set_list

def make_dir(path: Union[str, os.PathLike], exist_ok: bool=True) -> Union[str, os.PathLike]:
    os.makedirs(path, exist_ok=exist_ok)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--ts', type=int, required=False)
    
    args = parser.parse_args()
    print(f"n: {args.n}, ts: {args.ts}")
    
    ts: int = args.ts
    n: int = args.n
    
    real_n: int = 100
    out_dist_n: int = 12
    fake_n: int = 100
    
    max_mse_mean: str = 'max_mse_mean'
    max_mse_var: str = 'max_mse_var'
    
    output_fig_folder: str = make_dir(f'direct_record_fig_ts{ts}_n{n}', exist_ok=True)
    record_file: str = f'direct_record_ts{ts}_n{n}.safetensors'
    df_file: str = f'direct_record_df_ts{ts}_n{n}.hdf5'
    recorder: SafetensorRecorder = SafetensorRecorder()
    recorder.load(path=record_file, enable_update=False)
    
    # print(recorder.__data__)
    show_col_shapes(df=recorder.__data__)
    labels: torch.Tensor = recorder.__data__[SafetensorRecorder.LABEL_KEY]
    print(f"Unique Label: {labels.unique()}, Unique[0:{real_n * n}]: {labels[0:real_n * n].unique()}")
    print(f"Unique[{real_n * n}:{(real_n + out_dist_n) * n}]: {labels[real_n * n:(real_n + out_dist_n) * n].unique()}")
    print(f"Unique[{(real_n + out_dist_n) * n}:{(real_n + out_dist_n + fake_n) * n}]: {labels[(real_n + out_dist_n) * n:(real_n + out_dist_n + fake_n) * n].unique()}")
        
    recorder = reshape_recorder(recorder=recorder)
    
    print(f"After Reshape")
    show_col_shapes(df=recorder.__data__)
    
    df = load_cache_df(df_file)
    
    if df is None:
    
        real_list = []
        out_dist_list = []
        fake_list = []
        
        seqs = torch.split(recorder.__data__[SafetensorRecorder.SEQ_KEY], n)
        images = torch.split(recorder.__data__[SafetensorRecorder.IMAGE_KEY], n)
        reconsts = torch.split(recorder.__data__[SafetensorRecorder.RECONST_KEY], n)
        seq_mse_mean: list = []
        seq_mse_var: list = []
        seq_center_mse_mean: list = []
        seq_center_mse_var: list = []
        for i, (seq, image, reconst) in tqdm(enumerate(zip(seqs, images, reconsts))):
            assert len(torch.unique(image, dim=0)) == 1
            # seq_mse_mean.append(seq.mean(dim=0).squeeze())
            # seq_mse_var.append(seq.var(dim=0).squeeze())
            
            # print(f"image: {image.shape}, seq: {seq.shape}")
            mse_seq = mse_series(x0=image.to('cuda'), pred_orig_images=seq.to('cuda').transpose(0, 1)).cpu()
            center_mse_seq = center_mse_series(pred_orig_images=seq.to('cuda').transpose(0, 1)).cpu()
            
            seq_mse_mean.append(mse_seq.mean(dim=0).squeeze())
            seq_mse_var.append(mse_seq.var(dim=0).squeeze())
            
            seq_center_mse_mean.append(center_mse_seq.mean(dim=0).squeeze())
            seq_center_mse_var.append(center_mse_seq.var(dim=0).squeeze())

        seq_mse_mean = torch.stack(seq_mse_mean, dim=0)
        seq_mse_var = torch.stack(seq_mse_var, dim=0)
        seq_center_mse_mean = torch.stack(seq_center_mse_mean, dim=0)
        seq_center_mse_var = torch.stack(seq_center_mse_var, dim=0)
        # print(f"seq_mse_var: {seq_mse_var.shape}")
        # print(f"seq_mse_var.max(dim=1)[0]: {seq_mse_var.max(dim=1)[0].shape}")
        # df: pd.DataFrame = pd.DataFrame.from_dict(seq_mse_mean)
        df: pd.DataFrame = pd.DataFrame(seq_mse_mean.numpy())
        
        # col0, col999 = df.columns[0], df.columns[999]
        # print(df[df.columns].iloc[0].shape)
        # print(f"min: {min(df[df.columns].iloc[0])}")
        
        df['max_mse_mean'] = seq_mse_mean.max(dim=1)[0]
        df['max_mse_var'] = seq_mse_var.max(dim=1)[0]
        df['max_center_mse_mean'] = seq_center_mse_mean.max(dim=1)[0]
        df['max_center_mse_var'] = seq_center_mse_var.max(dim=1)[0]
        
        df['first_mse_mean'] = seq_mse_mean[:, 0].squeeze()
        df['first_center_mse_mean'] = seq_center_mse_mean[:, 0].squeeze()
        
        df['last_mse_mean'] = seq_mse_mean[:, -1].squeeze()
        df['last_mse_var'] = seq_mse_var[:, -1].squeeze()
        df['last_center_mse_mean'] = seq_center_mse_mean[:, -1].squeeze()
        df['last_center_mse_var'] = seq_center_mse_var[:, -1].squeeze()
        # df['label'] = recorder.__data__[SafetensorRecorder.LABEL_KEY]
        
        # print(list(df[:][0]))
        
        df['min'] = df[df.columns].min(axis=1)
        df['max'] = df[df.columns].max(axis=1)
        df['drop'] = df['first_mse_mean'] - df['min']
        df['increase'] = df['last_mse_mean'] - df['min']
        # df['label'] = ['Real'] * 100 + ['Out-Dist'] * 12 + ['Fake'] * 100
        
        # df.loc[:, df.columns] = df[df.columns].map(str)
        # df = df.rename(str, axis='columns')
    # else:
        # col0, col999 = df.columns[0], df.columns[999]
        # print(df[df.columns].iloc[0].shape)
        # print(f"min: {min(df[df.columns].iloc[0])}")
        
    save_cache_df(df=df, file=df_file)
    print(df)
    
    print(df['drop'])
    print(recorder.__data__[SafetensorRecorder.LABEL_KEY])
    
    # x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    # y_set_list = [df['increase'][:100], df['increase'][100:112], df['increase'][112:]]
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='increase', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Increment at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_incr_ts{ts}_n{n}'), xlabel='Drop', ylabel='Increase')
    
    # x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    # y_set_list = [df[col999][:100], df[col999][100:112], df[col999][112:]]
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='last_mse_mean', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & End at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_end_ts{ts}_n{n}'), xlabel='Drop', ylabel='End')
    
    # x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    # y_set_list = [df['max_var'][:100], df['max_var'][100:112], df['max_var'][112:]]
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='max_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Max Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_max_center_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Max Center MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_last_center_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Last Center MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='last_mse_mean', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_last_mse_mean_last_center_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Last Center MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='last_mse_mean', y_label='last_center_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Last Center MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_last_mse_mean_last_center_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Last Center MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Max MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_max_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Max MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='drop', y_label='last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Drop & Last MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_drop_last_mse_var_ts{ts}_n{n}'), xlabel='Drop', ylabel='Last MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='last_mse_mean', y_label='last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Last MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_last_mse_mean_last_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Last MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='max_mse_mean', y_label='max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Max MSE Mean & Max MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_max_mse_mean_max_mse_var_ts{ts}_n{n}'), xlabel='Max MSE Mean', ylabel='Max MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='max_mse_mean', y_label='last_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Max MSE Mean & Last MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_max_mse_mean_last_mse_var_ts{ts}_n{n}'), xlabel='Max MSE Mean', ylabel='Last MSE Var')
    
    x_set_list, y_set_list = x_y_set_list_gen(x_label='last_mse_mean', y_label='max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title=f"Last MSE Mean & Max MSE Var at Timestep: {ts} & {n} Samples", fig_name=os.path.join(output_fig_folder, f'scatter_last_mse_mean_max_mse_var_ts{ts}_n{n}'), xlabel='Last MSE Mean', ylabel='Max MSE Var')
    
    # x_set_list, y_set_list = x_y_set_list_gen(x_label=[i for i in range(ts)], y_label='max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    x_set_list, y_set_list = x_y_set_list_gen(x_label=df.columns[:ts], y_label='max_mse_var', real_n=real_n, out_dist_n=out_dist_n, fake_n=fake_n)
    plot_run(x_set_list=x_set_list, fig_name=os.path.join(output_fig_folder, f'line_direct_ts{ts}_n{n}.jpg'), title=f"Direct Reconstruction at Timestep: {ts} & {n} Samples", is_plot_var=False)
    
# %%
