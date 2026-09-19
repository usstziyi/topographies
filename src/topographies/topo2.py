import mne
import numpy as np
import matplotlib.pyplot as plt

# 中文标签需要指定字体，否则会显示成方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# MNE自带一个EEG示例数据集（需要联网首次下载）
from mne.datasets import sample

data_path = sample.data_path()

raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# 只读取EEG通道
raw = mne.io.read_raw_fif(raw_fname, preload=True)

print("挑选前通道数:", len(raw.ch_names))       # 例如 376
print("挑选前通道类型:", set(raw.get_channel_types()))  # {'meg', 'eeg', 'eog', 'stim', ...}

# 事件要从刺激通道 STI 014 提取，必须在挑选通道之前完成，
# 否则刺激通道被丢掉，后面 find_events 会报 Missing channels
events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
event_id = {'auditory/left': 1, 'auditory/right': 2}
print(f"事件数: {len(events)}")

raw.pick(picks="eeg", exclude="bads")

print("挑选后通道数:", len(raw.ch_names))       # 只剩 EEG，例如 60
print("挑选后通道类型:", set(raw.get_channel_types()))  # {'eeg'}


print(f"EEG通道数: {len(raw.ch_names)}")
print(f"采样率: {raw.info['sfreq']} Hz")


# 提取 10-20 秒的数据
raw_crop = raw.copy().crop(tmin=10, tmax=20)

# 带通滤波 1-40 Hz
raw.filter(1., 40., verbose=False)

bands = {
    'Delta (1-4 Hz)':   (1, 4),
    'Theta (4-8 Hz)':   (4, 8),
    'Alpha (8-13 Hz)':  (8, 13),
    'Beta (13-30 Hz)':  (13, 30),
    'Gamma (30-40 Hz)': (30, 40),
}

psd = raw_crop.compute_psd(fmin=1, fmax=40, method='welch')

fig = psd.plot_topomap(
    bands=bands,
    colorbar=True,
    # vlim='joint', # 共用一根色标尺子
    size=2,  # 未生效
    show=False
)

# # MNE 生成的 colorbar axes 通常在这些位置
# # 关掉除最后一个以外的所有 colorbar
# cbar_axes = [ax for ax in fig.axes if ax.get_label() == '<colorbar>']
# for ax in cbar_axes[:-1]:
#     ax.remove()

plt.savefig(
    'outputs/power_topomap.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',        # 背景白，避免透明
)