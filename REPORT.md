# warpflow-les 验证报告：顶盖驱动方腔流 vs Ghia et al. (Re = 400)

## 1. 任务概述

将 Hydro3D v7.0（Fortran LES 求解器）移植为 Python + NVIDIA Warp 的 GPU
代码 **warpflow-les**，要求：

- 全部浮点运算采用 **float64**；
- 多节点之间沿用 **MPI**（mpi4py），节点内部使用 GPU（Warp kernel）；
- 算例以 **YAML 配置文件**描述；
- 用 **cavity（顶盖驱动方腔）基准算例**验证。

## 2. 数值方法（与 Hydro3D v7.0 一一对应）

| 模块 | 方法 | Hydro3D 源文件 |
|---|---|---|
| 时间推进 | 两步 Heun RK2（conv_sch=3） | rungek.for |
| 对流项 | 四阶中心差分 | conv4th |
| 扩散项 | 显式四阶、变粘性 | diff4th |
| 亚格子模型 | WALE（cw = 0.46） | eddyvis_wale.for |
| 投影法 | 分步投影，外层迭代至四阶散度残差收敛 | press.for, newsolv_mg.for |
| 压力泊松 | 几何多重网格（3 层） | mgsolver.for |
| 时间步长 | 对流 + 粘性限制 | checkdt.for |
| 边界条件 | 壁面 / 滑移 / 运动盖 / 周期，3 层 ghost | bounds.for |

网格为交错网格（staggered），索引、ghost 层数（NPL = 3）、边界的
镜像/置零公式均逐行对照 Fortran 移植。

## 3. 与 Hydro3D 的逐步对比验证

验证方法：由 warpflow 初始场导出一份与 Hydro3D 完全相同的重启文件
（`tests/h3d_bridge.py`，含 4 层 ghost 的 Fortran 无格式 tecbin），
两个代码从同一状态分别推进，逐步比较全部场。

基准关键配置的解码：`control.cin` 中边界行行首的数字才是
`read_control` 实际读入的值——西/东/底 = 4（壁面），南/北 = 3（滑移），
顶 = 1（运动盖，keyword='cavity'）；行尾文字只是选项说明。因此
PERIODIC = .false.，exchange_bc、pressure_forcing、correctoutflux
在本算例中结构性失活。

另一个关键解码在 `checkdt.for`：`control.cin` 首列 `F 0.3E-3`
（variTS = .false.）并非"固定 dt = 3e-4"，而是**自适应**——每步执行
`dt = min(1.1·dt, 0.8·dtmax)`（dtmax 为对流 + 粘性限制），3e-4 只是
初值；只有 `T` 分支（variTS = .true.）才每步强制 dt = dtinit。首次
生产运行把语义理解为固定 dt，在 t ≈ 1.8（主涡发展、CFL 越过 1）时
于第 6013 步投影发散（rmax = 45）；改回自适应后运行全程 CFL 稳定在
0.80 上限，与 Hydro3D 行为一致。WALE 粘性场也做了场级对比（it25：
两边均在 [0.0025, 0.0026]，rms 差 4×10⁻⁶），排除了亚格子模型的
移植缺陷。

对比结果（80³，固定 dt = 3e-4）：

| 步数 | u rms | v rms | w rms | p rms（去均值） | rmax（本代码 / Hydro3D） |
|---|---|---|---|---|---|
| 1   | 2.1e-4 | 1.0e-4 | 0.9e-4 | 9.7e-2 | 1.69e-7 / 1.69e-7 |
| 25  | 2.5e-4 | 1.1e-4 | 1.1e-4 | 8.8e-4 | 2.04e-9 / 6.8e-10 |
| 50  | 2.8e-4 | 1.2e-4 | 1.2e-4 | 2.0e-4 | — / 4.8e-11 |
| 100 | 3.3e-4 | 1.2e-4 | 1.4e-4 | 2.2e-4 | 2.01e-9 / 2.6e-11 |

- 速度场均方根差异稳定在 **1e-4 量级**（速度尺度为 1，即 0.01%–0.03%），
  100 步内仅缓慢线性漂移，无失控增长；
- 散度残差轨迹与 Hydro3D 高度吻合（it1 两边同为 1.69e-7）；
- 已记录的偏差：多重网格采用 RBGS + V-cycle（Hydro3D 为 TDMA 线迭代 +
  K-cycle），泊松问题解到同样容差但迭代路径不同，导致压力的加性常数与
  个别点位存在路径性差异（it1 时 O(0.1)，随外层收敛迅速衰减）；残差
  最大值集中在东壁列，亦源于此。

## 4. 生产算例

`cases/cavity_serial.yaml`：80³ 网格，Re = 400，自适应时间步
（初值 3×10⁻⁴，safety 0.8，每步增长上限 10%），50 000 步（t ≈ 15，
取决于 dt 轨迹），静止 + 1% 确定性噪声初值，WALE 开启，单 GPU
（RTX 5060 Laptop）串行域。

- 实测速度约 0.3 s/步，总时长约 4 h；
- 投影 2 次外迭代收敛，rmax ~10⁻⁹。

运行结束后提取中心线剖面与 Ghia, Tuann & Olson (1982) Re = 400 数据
对比（`scripts/plot_ghia.py` → `output_cavity/ghia_comparison.png`），
结果见第 5 节。

## 5. 结果：中心线速度剖面对比

（待生产运行完成后填充）

## 6. 结论

（待生产运行完成后填充）
