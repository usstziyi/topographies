import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import sample

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)

# 1) 仍然要先从刺激通道提取事件（annotations 的来源）
events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
print(events.shape) # (n_events, 3)
event_id = {'auditory/left': 1, 'auditory/right': 2} # events_id
event_desc = {1: 'auditory/left', 2: 'auditory/right'} # annotations事件描述

# 2) 只保留我们关心的两类事件，映射成描述名
# np.isin(a, b) 逐个检查 a 中每个元素是否出现在 b 里，返回同长度的布尔数组
# np.isin([1, 1, 2, 3, 2, 1, 4, 2], [1, 2])
# [True, True, True, False, True, True, False, True]
events_selected = events[np.isin(events[:, 2], [1, 2])]
print(events_selected.shape) # (n_events_selected, 3)
annot = mne.annotations_from_events(
    events_selected,
    sfreq=raw.info['sfreq'],
    event_desc=event_desc,
)

# 打印Annotations对象的常用可查看属性
print(f"Annotations 总数量: {len(annot)}")
print(f"Annotations 触发时间（onset，单位：秒）:\n{annot.onset}")
print(f"Annotations 持续时间（duration，单位：秒）:\n{annot.duration}")
print(f"Annotations 事件描述（description）:\n{annot.description}")
print("\nAnnotations 对象详细概览:")
print(annot)



raw.set_annotations(annot)          # 挂到 raw 上

# 3) 挑 EEG（在 annotations 之后、建 epochs 之前都行）
raw.pick(picks="eeg", exclude="bads")

# 4) 带通滤波
raw.filter(1., 40., verbose=False)

# 5) 用 annotations 建 epochs：event_id 用字符串描述
epochs = mne.Epochs(
    raw,
    events=None,                    # 关键：不给 events
    event_id=event_id,              # {'auditory/left': 1, ...} 这里 value 只是占位
    tmin=-0.2, tmax=0.5,
    baseline=(-0.2, 0),
    preload=True, verbose=False,
)
print(f"epochs 数量: {len(epochs)}")
print(f"epochs 形状: {epochs.get_data().shape}")

evoked = epochs['auditory/left'].average()

times_to_plot = np.linspace(0.05, 0.35, 7)
fig = evoked.plot_topomap(
    times=times_to_plot, ch_type='eeg', cmap='RdBu_r',
    size=1, time_unit='ms', vlim=(-10, 10), show=False,
)
fig.suptitle('听觉诱发EEG地形图（左耳刺激）', fontsize=13)
fig.savefig('outputs/evoked_topomap_times.png', dpi=150,
            bbox_inches='tight', facecolor='white')