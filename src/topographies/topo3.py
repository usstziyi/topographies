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

# 带通滤波 1-40 Hz
# 注意：滤波要放在建 epochs 之前，epoched / evoked 数据才会带上滤波结果
raw.filter(1., 40., verbose=False)

# 用 Evoked 对象绘制（基于事件相关电位）
# events / event_id 已在上面从刺激通道 STI 014 提取（挑选通道之前）

# 构建 epochs,多个试次（trial）的集合
epochs = mne.Epochs(
    raw, 
    events, 
    event_id, 
    tmin=-0.2, 
    tmax=0.5,
    baseline=(-0.2, 0), 
    preload=True, 
    verbose=False
)

print(f"epochs 数量: {len(epochs)}")
print(f"epochs 形状: {epochs.get_data().shape}")     # (n_epochs, n_channels, n_times)


# 平均得到 evoked,多个试次的平均结果
evoked = epochs['auditory/left'].average() # (n_channels, n_times)

# 绘制不同时间点的地形图序列
# 直接用 Evoked.plot_topomap，MNE 会按 times 自动排子图，不用手写循环
times_to_plot = np.linspace(0.05, 0.35, 7)



fig = evoked.plot_topomap(
    times=times_to_plot,
    ch_type='eeg',
    cmap='RdBu_r',
    size=1,     # 生效
    time_unit='ms',
    vlim=(-10, 10),  # 手动指定对称范围单位：uV
    show=False,
)

fig.suptitle('听觉诱发EEG地形图（左耳刺激）', fontsize=13)

fig.savefig(
    'outputs/evoked_topomap_times.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',        # 背景白，避免透明
)


