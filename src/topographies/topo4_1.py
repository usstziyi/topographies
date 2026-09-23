import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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


# ============ 自建交互面板（波形 + 游标 + 联动地形图）============
# 为什么不用 evoked.plot_topomap(times='interactive')：
#   它只在滑块那条 axes 上顺带画了波形，而那条 axes 的 y 轴被滑块自己的元素
#   顶到 0~1 量级，波形是 V 量级（~1e-5），全被压成贴底的一条直线，看不见。
#
# 所以这里自己搭三层：
#   第 1 层 ax_topo   —— 单个时刻的地形图
#   第 2 层 ax_wave   —— 所有通道波形 + 一条竖直游标
#   第 3 层 ax_slider —— 时间滑块
#   拖动滑块 → 游标跟着走 + 重画该时刻的地形图

# 波形轴的单位换算：evoked.data 是 V，为了 y 轴刻度可读，画图时统一换成 uV
# （地形图不用自己换算，evoked.plot_topomap 内部会按 scalings 处理）
data_uv = evoked.data * 1e6
times_ms = evoked.times * 1000

t_init = 0.0  # 初始时刻（秒）：刺激 onset

fig = plt.figure(figsize=(8, 7))
# 创建一个 3 行 1 列 的网格布局，第 1 行是地形图，第 2 行是波形，第 3 行是滑块
gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.8, 0.35], hspace=0.45)

ax_topo = fig.add_subplot(gs[0])
ax_wave = fig.add_subplot(gs[1])
ax_slider = fig.add_subplot(gs[2])

# --- 第 2 层：波形 + 游标 ---
ax_wave.plot(times_ms, data_uv.T, color='k', lw=0.4, alpha=0.5)
ax_wave.axvline(0, color='gray', ls='--', lw=1)              # 刺激 onset 参考线
cursor = ax_wave.axvline(t_init * 1000, color='r', lw=1.5)   # 当前时刻游标
ax_wave.set_xlim(times_ms[0], times_ms[-1])
ax_wave.set_xlabel('Time (ms)')
ax_wave.set_ylabel('Amplitude (uV)')
ax_wave.set_title('所有EEG通道波形（灰虚线=刺激onset，红线=当前时刻）', fontsize=10)


def _update(t):
    """滑块回调：把游标移到时刻 t（秒），并重画该时刻的地形图。"""
    idx = int(np.argmin(np.abs(evoked.times - t)))
    t_actual = evoked.times[idx]

    # 游标横坐标单位是 ms，和上面波形的横轴保持一致
    cursor.set_xdata([t_actual * 1000, t_actual * 1000])

    # 每个时刻都是重建 artists，最省事、也不会残留上一个时刻的等值线
    # 用 evoked.plot_topomap 的两个好处：
    #   1) vlim 按 uV 处理，MNE 内部按 scalings 从 V 自动换算，不用自己 *1e6
    #   2) 自动把时刻写进 axes 标题（如 "120 ms"），不用自己 set_title
    # 注意：一旦传了 axes，colorbar 就必须是 False，
    #       否则 MNE 要求 axes 个数 = 时刻个数 + 1 个 colorbar 位
    ax_topo.clear()
    evoked.plot_topomap(
        times=[t_actual],
        ch_type='eeg',
        cmap='RdBu_r',
        contours=6,
        vlim=(-10, 10),      # 固定色标范围，单位 uV，拖动时各时刻才能直接对比
        size=1,
        time_unit='ms',
        time_format='%.0f ms',   # MNE 默认是 "%01d ms"（截断），这里改成四舍五入
        axes=ax_topo,
        colorbar=False,
        show=False,
    )
    # 不用再 fig.canvas.draw_idle()：
    # 传了 axes 时 evoked.plot_topomap 结尾自己会 fig.canvas.draw()（MNE 源码里那行），
    # 而游标在它之前就移好了，会被同一次重绘一起带出去


# 先画初始时刻，再从它在地形图 axes 上留下的 image 建 colorbar；
# 后续更新只是 clear + 重画 ax_topo，colorbar 是独立 axes，不受影响
_update(t_init)
cbar = fig.colorbar(ax_topo.images[0], ax=ax_topo, fraction=0.046, pad=0.04)
cbar.set_label('Amplitude (uV)')

# --- 第 3 层：时间滑块 ---
slider = Slider(
    ax_slider,
    'Time (s)',
    evoked.times[0],
    evoked.times[-1],
    valinit=t_init,
    valfmt='%1.3f',
)
slider.on_changed(_update)

fig.suptitle('EEG地形图（拖动滑块联动查看）', fontsize=13)

# 交互窗口靠 plt.show() 保持，关掉窗口后再保存图片
plt.show()

fig.savefig(
    'outputs/evoked_topomap_linked.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',        # 背景白，避免透明
)
