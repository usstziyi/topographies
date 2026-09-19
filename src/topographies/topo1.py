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

# # 取 10-20 秒的数据
# start, stop = 10, 20
# result = raw[:, int(start * raw.info['sfreq']):int(stop * raw.info['sfreq'])]
# data = result[0] # 数据矩阵
# times = result[1] # 时间轴

# 提取 10-20 秒的数据
raw_crop = raw.copy().crop(tmin=10, tmax=20)

# 带通滤波 1-40 Hz
raw.filter(1., 40., verbose=False)

# 设置电极蒙太奇（如果原始数据没有标准位置）
# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage)

# 看有没有电极位置
# print("digitization points:", raw.info['dig'])      # 有内容
# print("montage:", raw.get_montage())                 # 已有 montage



# 计算某频段功率（例如 alpha 8-13 Hz）
# 得到的是这个频段内每个频率点的 PSD 值，单位为 V²/Hz
alpha_power = raw_crop.compute_psd(fmin=8, fmax=13, method='welch',n_fft= 4096)


# 绘制功率谱密度（PSD）曲线图
# average=True：把所有 EEG 通道平均成一条线；ci='sd'：画出 ±1 标准差阴影带；color='red'：线条颜色
fig = alpha_power.plot(show=False)

# 手动设置坐标轴范围：fig.axes[0] 是主坐标轴（后面可能还有 colorbar 轴）
ax = fig.axes[0]
ax.set_xlim(8, 13)       # 横轴范围受限于 compute_psd 的 fmin/fmax，只能在这段内缩放

plt.savefig(
    fname='outputs/alpha_psd.png', 
    dpi=150,
    facecolor='white',        # 背景白，避免透明
    bbox_inches='tight',
)


# average=True：把所有 EEG 通道平均成一条线；ci='sd'：画出 ±1 标准差阴影带；color='red'：线条颜色
fig = alpha_power.plot(average=True, ci='sd', color='red', show=False)

# 手动设置坐标轴范围：fig.axes[0] 是主坐标轴（后面可能还有 colorbar 轴）
ax = fig.axes[0]
ax.set_xlim(8, 13)       # 横轴范围受限于 compute_psd 的 fmin/fmax，只能在这段内缩放

plt.savefig(
    fname='outputs/alpha_avg_psd.png', 
    dpi=150,
    facecolor='white',        # 背景白，避免透明
    bbox_inches='tight',
)



# 画地形图
# 注意：alpha_power 只包含 8-13 Hz，必须显式指定频段；
# 否则 plot_topomap 默认要画 Delta/Theta 等频段，会因该频段无频点而报错
# 方式一：自己建 axes 再传进去（推荐，size 才能真正生效） 
fig, ax = plt.subplots(figsize=(6 ,6))
alpha_power.plot_topomap(
    bands={ 'Alpha (8-13 Hz)' : ( 8 , 13 )},
    axes=ax,
    size=2, # 未生效
    cmap= 'Reds' ,
    show= False ,
)

fig.savefig(
    fname='outputs/alpha_topomap.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',        # 背景白，避免透明
)



# # 手动绘制地形图
# psds, freqs = alpha_power.get_data(return_freqs=True)

# print(psds.shape)
# print(freqs.shape)
# print(freqs)


# # 对频率维求平均，得到每个通道的平均功率
# psd_mean = psds.mean(axis=1)  # shape: (n_channels,)


# # mne.viz.plot_topomap 返回 (im, cn) 元组，不是 Figure，
# # 所以要先自己建好 fig/ax 再传进去，才能用 fig.savefig 保存
# fig, ax = plt.subplots(figsize=(5, 5))

# # 手动绘制地形图
# mne.viz.plot_topomap(
#     psd_mean,
#     raw.info,  # 传递原始数据的 info，包含通道位置信息
#     axes=ax,
#     cmap='Reds', 
#     contours=6,
#     sensors=True,          # 显示电极点
#     show=False
# )



# ax.set_title('Alpha频段 (8-13Hz) 功率地形图', fontsize=12)
# plt.colorbar(ax.images[0], ax=ax, label='Power (V²/Hz)')
# plt.tight_layout()

# fig.savefig(
#     'outputs/alpha_topomap_viz.png',
#     dpi=150,
#     bbox_inches='tight',
#     facecolor='white',        # 背景白，避免透明
# )

