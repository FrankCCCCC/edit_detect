# %%
import pandas as pd
import torch

from detect import plot_scatter
from util import DirectRecorder, TensorDict, SafetensorRecorder

if __name__ == "__main__":
    ts: int = 50
    n: int = 2
    record_file: str = f'/workspace/research/edit_detect/direct_record_ts{ts}_n{n}.safetensors'
    recorder: SafetensorRecorder = SafetensorRecorder()
    recorder.load(path=record_file, enable_update=False)
    
    # print(recorder.__data__)
    for key, val in recorder.__data__.items():
        print(f"[{key}]: {val.shape}")
        
    recorder.__data__[SafetensorRecorder.IMAGE_KEY] = recorder.__data__[SafetensorRecorder.IMAGE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.NOISE_KEY] = recorder.__data__[SafetensorRecorder.NOISE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = recorder.__data__[SafetensorRecorder.NOISY_IMAGE_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.RECONST_KEY] = recorder.__data__[SafetensorRecorder.RECONST_KEY].reshape(-1, 3, 32, 32)
    # recorder.__data__[SafetensorRecorder.LABEL_KEY] = recorder.__data__[SafetensorRecorder.LABEL_KEY].reshape(-1, 3, 32, 32)
    recorder.__data__[SafetensorRecorder.SEQ_KEY] = recorder.__data__[SafetensorRecorder.SEQ_KEY].reshape(-1, ts)
    recorder.__data__[SafetensorRecorder.TS_KEY] = recorder.__data__[SafetensorRecorder.TS_KEY].reshape(-1, ts)
    
    print(f"After Reshape")
    for key, val in recorder.__data__.items():
        print(f"[{key}]: {val.shape}")
    
    real_list = []
    out_dist_list = []
    fake_list = []
    
    seqs = torch.split(recorder.__data__[SafetensorRecorder.SEQ_KEY], n)
    images = torch.split(recorder.__data__[SafetensorRecorder.IMAGE_KEY], n)
    seq_mean: list = []
    seq_var: list = []
    for i, (seq, image) in enumerate(zip(seqs, images)):
        assert len(torch.unique(image, dim=0)) == 1
        seq_mean.append(seq.mean(dim=0).squeeze())
        seq_var.append(seq.var(dim=0).squeeze())

    # print(f"torch.stack(seq_var, dim=0): {torch.stack(seq_var, dim=0).shape}")
    # print(f"torch.stack(seq_var, dim=0).max(dim=1)[0]: {torch.stack(seq_var, dim=0).max(dim=1)[0].shape}")
    df: pd.DataFrame = pd.DataFrame.from_dict(torch.stack(seq_mean, dim=0))
    df['max_var'] = torch.stack(seq_var, dim=0).max(dim=1)[0]
    # df['label'] = recorder.__data__[SafetensorRecorder.LABEL_KEY]
    
    # print(list(df[:][0]))
    col0, col999 = df.columns[0], df.columns[-1]
    print(df[df.columns].iloc[0].shape)
    print(f"min: {min(df[df.columns].iloc[0])}")
    
    df['min'] = df[df.columns].min(axis=1)
    df['max'] = df[df.columns].max(axis=1)
    df['drop'] = df[col0] - df['min']
    df['increase'] = df[col999] - df['min']
    df['label'] = ['Real'] * 100 + ['Out-Dist'] * 12 + ['Fake'] * 100
    
    print(df)
    
    print(df['drop'])
    print(recorder.__data__[SafetensorRecorder.LABEL_KEY])
    
    x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    y_set_list = [df['increase'][:100], df['increase'][100:112], df['increase'][112:]]
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title="Drop & Increment", fig_name=f'scatter_drop_incr_ts{ts}_n{n}', xlabel='Drop', ylabel='Increase')
    
    x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    y_set_list = [df[col999][:100], df[col999][100:112], df[col999][112:]]
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title="Drop & End", fig_name=f'scatter_drop_end_ts{ts}_n{n}', xlabel='Drop', ylabel='End')
    
    x_set_list = [df['drop'][:100], df['drop'][100:112], df['drop'][112:]]
    y_set_list = [df['max_var'][:100], df['max_var'][100:112], df['max_var'][112:]]
    plot_scatter(x_set_list=x_set_list, y_set_list=y_set_list, title="Drop & Max Var", fig_name=f'scatter_drop_max_var_ts{ts}_n{n}', xlabel='Drop', ylabel='Max Var')
    
# %%
