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
evoked = epochs['auditory/left'].average()  # (n_channels, n_times)



# times='interactive' 是 MNE 的专用关键字：
#   上面一张地形图 + 下面一条 matplotlib Slider 时间滑块
#   拖动滑块即可实时重画该时刻的地形图
#   注：MNE 虽然在源码里把 evoked 波形也画在了滑块那条 axes 上，
#       但两者量纲不匹配（波形是 V 量级，滑块 axes 的 y 轴是 0~1 量级），
#       波形被压成贴底的一条直线，实际看不到，别指望它当"波形+游标"用
#
# 注意：
#   1) 交互模式下不能再传 axes，nrows / ncols 也会被忽略
#   2) vlim 必须固定（不能让它随时刻自动缩放），否则拖动时色标一直在变，
#      不同时刻没法直接对比
#   3) 需要可交互的 matplotlib 后端；若在 Jupyter 里不出窗口，先执行 %matplotlib qt
fig = evoked.plot_topomap(
    times='interactive',
    ch_type='eeg',
    cmap='RdBu_r',
    contours=6,      # 等值线条数
    size=1,
    time_unit='ms',
    vlim=(-10, 10),  # 固定色标范围，单位：uV，保证各时刻可比
    show=True,       # 必须 True，窗口才会弹出来并保持可交互
)

fig.suptitle('听觉诱发EEG地形图（左耳刺激，拖动滑块查看不同时刻）', fontsize=13)

# 交互窗口靠 plt.show() 保持，关掉窗口后再保存图片
fig.savefig(
    'outputs/evoked_topomap_interactive.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',        # 背景白，避免透明
)
