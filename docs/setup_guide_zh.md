# Vec-QMDP 工程配置与运行指南

基于 README 和 `docs/` 中所有文档的完整内容，以下是从零开始配置和运行的详尽步骤。

> 本文档对应目录结构：
> ```
> ~/Desktop/workspace/e2e_workspace/
> ├── dataset/
> │   ├── maps/
> │   │   ├── nuplan-maps-v1.0.json
> │   │   ├── sg-one-north/
> │   │   ├── us-ma-boston/
> │   │   ├── us-nv-las-vegas-strip/
> │   │   └── us-pa-pittsburgh-hazelwood/
> │   └── nuplan-v1.1/
> │       └── mini/           ← mini 数据集的 .db 文件
> ├── exp/
> │   └── simulation/
> ├── nuplan-devkit/
> └── VecQMDP/
> ```

---

## 阶段一：Conda 环境与系统依赖

### 1.1 创建并激活 conda 环境

```bash
conda create -n vec_qmdp python=3.9
conda activate vec_qmdp
```

### 1.2 安装 Eigen3（通过 conda，CMake 构建时会自动通过 `$CONDA_PREFIX` 发现）

```bash
conda install -c conda-forge eigen
```

### 1.3 安装系统级 Boost 和 OpenGL

```bash
sudo apt-get install -y libboost-all-dev libgl1-mesa-glx
```

> **注意**：Boost.Python 必须链接到 Python 3.9。如果你的系统 Boost 版本对应的 Python 不是 3.9，需要从源码编译 Boost 并指定 `--with-python=3.9`。可以用 `dpkg -l | grep libboost-python` 查看已安装的版本。

### 1.4 安装 cmake（如果系统没有或版本低于 3.15）

```bash
pip install cmake
# 或者
conda install cmake
```

---

## 阶段二：Python 依赖

### 2.1 安装 nuPlan devkit

```bash
cd ~/Desktop/workspace/e2e_workspace/nuplan-devkit
pip install -e .
cd ~/Desktop/workspace/e2e_workspace/VecQMDP
```

### 2.2 安装 VecQMDP Python 依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 顶行 `--find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html` 是为 CUDA 12.4 配置的。驱动支持 CUDA 13.0 时向下兼容，不影响。如果安装报错，可以先单独装 PyTorch：
> ```bash
> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```
> 然后再 `pip install -r requirements.txt`。

### 2.3 修复 bokeh + NumPy 2.x 兼容性（一次性）

NumPy 2.x 移除了 `np.bool8`，需要打补丁：

```bash
sed -i 's/bokeh_bool_types += (np.bool8,)/bokeh_bool_types += (np.bool_,)/' \
  "$(python -c 'import site; print(site.getsitepackages()[0])')/bokeh/core/property/primitive.py"
```

### 2.4 安装 QCNet 预测模型

QCNet 是 PyTorch 的多模态轨迹预测模型，被 `python_planner/vec_qmdp_planner_module.py` 导入用于生成周围车辆的未来轨迹 belief：

```bash
cd ~/Desktop/workspace/e2e_workspace/VecQMDP/python_planner/qcpredictor
pip install -e .
cd ~/Desktop/workspace/e2e_workspace/VecQMDP
```

### 2.5 验证 PyTorch GPU 可用

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

应输出 `CUDA available: True, Device: NVIDIA GeForce RTX 3080 Ti`。

---

## 阶段三：构建 C++ 组件

### 3.1 构建 GEOS 静态库（仅首次）

C++ 碰撞检测模块（`STRtree` + `OccupancyMap`）需要带 AVX 支持的 GEOS 静态库：

```bash
cd ~/Desktop/workspace/e2e_workspace/VecQMDP/external/geos
bash geos_with_simd.sh
cd build-avx
make install
cd ~/Desktop/workspace/e2e_workspace/VecQMDP
```

这会在 `~/.local/lib/` 生成 `libgeos.a` 和 `libgeos_c.a`。`CMakeLists.txt` 中 `GEOS_ROOT` 默认指向 `$ENV{HOME}/.local`，所以不需要改。

### 3.2 构建 VecQMDP 共享库

确保 conda 环境已激活（构建脚本依赖 `$CONDA_PREFIX` 来定位 Eigen 和 Python）：

```bash
conda activate vec_qmdp
bash scripts/build_vec_qmdp.sh --opt=O3
```

构建过程：

1. CMake 配置 → 发现 Eigen3、Boost.Python、GEOS、Python
2. 编译四个 OBJECT 库（`vec_qmdp_core`、`vec_qmdp_planning`、`vec_qmdp_utils`、`vec_qmdp_collision`）
3. 链接为静态库 `libvec_qmdp.a`
4. 链接为共享库 `vec_qmdp_closed_planner.so`（包含 Boost.Python 绑定）
5. 将 `.so` 复制到 `python_planner/`

验证构建：

```bash
cd python_planner
python -c "import vec_qmdp_closed_plainer; print('Shared library loaded successfully')"
cd ..
```

如果构建失败，用 debug 模式查看详细日志：

```bash
bash scripts/build_vec_qmdp.sh --clean --debug --opt=O3
```

其他构建选项（参见 `scripts/build_vec_qmdp.sh`）：

```bash
bash scripts/build_vec_qmdp.sh -h                    # 查看所有选项
bash scripts/build_vec_qmdp.sh --clean --opt=O3      # 清理后重新构建
bash scripts/build_vec_qmdp.sh --asan                 # AddressSanitizer 内存检查
bash scripts/build_vec_qmdp.sh --perf                 # 性能 profiling 支持
bash scripts/build_vec_qmdp.sh --rebuild-geos         # 同时重建 GEOS
```

---

## 阶段四：环境变量配置

创建 `env.sh`（不在 `~/.zshrc` 中永久添加）：

```bash
cat > ~/Desktop/workspace/e2e_workspace/env.sh << 'EOF'
export NUPLAN_DATA_ROOT="$HOME/Desktop/workspace/e2e_workspace/dataset"
export NUPLAN_MAPS_ROOT="$HOME/Desktop/workspace/e2e_workspace/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/Desktop/workspace/e2e_workspace/exp"
export NUPLAN_DEVKIT_ROOT="$HOME/Desktop/workspace/e2e_workspace/nuplan-devkit"
export NUPLAN_HYDRA_CONFIG_PATH="$HOME/Desktop/workspace/e2e_workspace/nuplan-devkit/nuplan/planning/script/config"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="$HOME/Desktop/workspace/e2e_workspace/VecQMDP/python_planner/qcpredictor:$HOME/Desktop/workspace/e2e_workspace/nuplan-devkit:$HOME/Desktop/workspace/e2e_workspace/VecQMDP/python_planner:$HOME/Desktop/workspace/e2e_workspace/VecQMDP"
EOF
```

每次开新终端时：

```bash
source ~/Desktop/workspace/e2e_workspace/env.sh
conda activate vec_qmdp
```

验证环境变量：

```bash
env | grep -E "NUPLAN|PYTHONPATH"
```

---

## 阶段五：为 mini 数据集创建 Scenario Filter

默认的 `val14_split` / `test_hard` / `test14_random` 引用的 scenario token 来自完整数据集，mini 数据集中不存在。你需要创建自己的 filter：

```bash
cat > ~/Desktop/workspace/e2e_workspace/VecQMDP/python_planner/scripts/config/common/scenario_filter/mini_split.yaml << 'EOF'
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: 'all'

scenario_types:
  - following_lane_with_lead
  - stopping_with_lead
  - starting_left_turn
  - starting_right_turn
  - changing_lane

scenario_tokens: null
limit_total_scenarios: 10
randomize_sample_order: true
EOF
```

然后修改 `scripts/simulation/sim_planner.py`，在 `VALID_SPLITS` 中加入 `mini_split`：

```python
VALID_SPLITS = ['val14_split', 'test_hard', 'test14_random', 'mini_split']
```

---

## 阶段六：运行 nuPlan 闭环仿真

```bash
source ~/Desktop/workspace/e2e_workspace/env.sh
conda activate vec_qmdp
cd ~/Desktop/workspace/e2e_workspace/VecQMDP

# 顺序模式（调试用，一次跑一个 scenario）
python scripts/simulation/sim_planner.py \
  --split mini_split \
  --challenge closed_loop_nonreactive_agents \
  --worker sequential
```

### 数据流（来自 `docs/architecture.md` 和 `docs/qmdp_trajectory_planner.md`）

```
Python 传入感知数据 + HD Map
  → QCNet 预测周围车辆的多模态轨迹 (NetBelief)
  → QMDPTrajectoryPlanner::planTrajectory()
    → filterDiscardedAgents() 三阶段过滤
        (Frenet空间 → 轨迹交叉 → 碰撞排除)
    → VecQMDP_AD::beliefTreeSearch() 多线程搜索
        → exploreNodesVote() 多数投票深度同步 UCB
        → expandNodesBatch()
            → ContextQMDP::StepBatch() SIMD 前向仿真
                (IDM纵向 + Stanley横向 + STRtree+SAT碰撞)
        → backPropagate() Q值回传 + 剪枝
    → getBestAction() 带迟滞偏好的动作选择
    → TrajectoryOptimization::optimize() 三阶段轨迹优化
        → importanceSampleScenarios() 重要性采样
        → generateProposalTrajectory*Batch() 0.1s分辨率提案
        → Tracker::track() LQR 跟踪
        → crossScenarioEvaluationBatch() 跨场景评估
        → checkAndGenerateEmergencyBrake() 安全制动
    → 返回最优轨迹
```

### 仿真参数说明

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--split` | `val14_split` | `mini_split`, `val14_split`, `test_hard`, `test14_random` | 场景数据划分 |
| `--challenge` | `closed_loop_nonreactive_agents` | `closed_loop_nonreactive_agents`, `closed_loop_reactive_agents` | 仿真模式（reactive 需要 `reactive` 分支） |
| `--worker` | `ray_distributed` | `ray_distributed`, `sequential` | `sequential` 适合调试，`ray_distributed` 适合批量评估 |

---

## 阶段七（可选）：运行 RockSample 基准测试

不需要 nuPlan 数据集，用于理解搜索引擎核心。详见 `docs/tutorial.md`。

### 7.1 启用 Homogeneous Search

编辑 `include/planning/vec_qmdp_dynamic.hpp`，取消注释：

```cpp
#define ENABLE_HOMOGENOUS_SEARCH
```

> **仅限 RockSample 使用**；nuPlan 评估时必须注释掉此宏。Homogeneous Search 让所有场景共享同一条搜索路径（只有 scenario-0 做 UCB），适用于场景动态一致且 expansion 开销低的领域（如 RockSample）。对于自动驾驶等异构场景，必须使用 Heterogeneous Search（默认模式）以保持 belief 多样性。参见 `docs/vec_qmdp.md` 第 1.2.3 节。

### 7.2 构建并运行

```bash
# 构建 VecQMDP（动态后端，推荐）
bash examples/rock_sample/build_vec_qmdp.sh --clean --opt=O3

# 或使用静态后端（A^H < ~100,000 时更快）
# bash examples/rock_sample/build_vec_qmdp.sh --static-solver

# 构建 DESPOT 基线
bash examples/rock_sample/build_despot.sh

# 运行基准测试（各 20 次仿真）
bash examples/rock_sample/benchmark/run_vecqmdp.sh
bash examples/rock_sample/benchmark/run_despot.sh

# 分析对比结果
python3 examples/rock_sample/benchmark/analyze.py
```

### 7.3 运行完记得恢复

把 `ENABLE_HOMOGENOUS_SEARCH` 重新注释掉，再构建：

```bash
bash scripts/build_vec_qmdp.sh --opt=O3
```

---

## 关键参数调整

根据 `docs/utils.md`，以下参数在 `include/utils/params.hpp` 中定义，修改后需重新构建：

### 规划核心

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `MAX_PLANNING_TIME` | 8.0f | 每帧规划时间预算(秒) |
| `NUM_THREADS` | 8 | 搜索线程数 |
| `NUM_SCENARIOS_PER_THREAD` | 8 | 每线程场景数（必须是8的倍数） |
| `TREE_HEIGHT` | 4 | 搜索树深度 |
| `NUM_ACTIONS` | 9 | 动作空间 = 3路径 × 3偏移 |
| `NUM_SCENARIOS` | 64 | 总 belief 场景数 |
| `MAX_SIM_VEHICLES` | 96 | 最大模拟车辆数（对齐到8的倍数） |

### 时间与离散化

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `TIME_STEP` | 0.2s | 仿真时间步长 |
| `STEP_TIME` | 2.0s | 树搜索中每层动作持续时间 |
| `ROLLOUT_TIME` | 2.0s | 树外 rollout 展望时间 |
| `LOOKAHEAD_TIME` | 4.8s | 前向预测窗口 |

### 奖励与惩罚

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `CRASH_PENALTY` | -1000 | 碰撞惩罚 |
| `MOVEMENT_PENALTY` | 5.0 | 速度偏差权重 |
| `ACTION_PENALTY` | -2.0 | 动作切换惩罚（防抖动） |
| `PRUNED_THRESHOLD` | -1000 | 剪枝阈值 |
| `MAX_VEL` | 15.0f | 最大速度(m/s) |

### 提前终止

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `EARLY_TERM_CHECK_INTERVAL` | 5 | 每N次迭代检查收敛 |
| `EARLY_TERM_MIN_EXPAND_CALLS` | 30 | 最少展开次数后才开始检查 |
| `EARLY_TERM_STABLE_ITERATIONS` | 5 | 要求连续稳定迭代数 |
| `EARLY_TERM_Q_CHANGE_THRESHOLD` | 0.05 | Q值相对变化阈值 |

### 不同 split 推荐配置

| 分支 | 划分 | `MAX_PLANNING_TIME` | `NUM_THREADS` | `NUM_SCENARIOS_PER_THREAD` |
|------|------|---------------------|---------------|---------------------------|
| `main` (NR) | `val14_split` | 8.0f | 8 | 8 |
| `main` (NR) | `test_hard` | 8.0f | 8 | 8 |
| `main` (NR) | `test14_random` | 0.5f | 8 | 8 |
| `reactive` (R) | 所有划分 | 8.0f | 8 | 8 |

---

## 常见问题排查

### CPU 不支持 AVX2

至少需要 Intel Haswell 或 AMD Zen 1。检查方法：`lscpu | grep avx2`

### CMake 找不到 Eigen3

确保 `conda install -c conda-forge eigen` 已在当前环境中安装，且 `CMAKE_PREFIX_PATH` 包含 `$CONDA_PREFIX`（构建脚本会自动设置）。

### Boost.Python 版本不匹配

如果系统 Boost 编译时链接的不是 Python 3.9，需要从源码重新编译 Boost。或者检查：`dpkg -l | grep libboost-python`。

### GPU / CUDA 问题

`nvcc --version` 找到与否不影响运行——PyTorch 自带 CUDA 运行时。只需 `nvidia-smi` 显示驱动正常且 `torch.cuda.is_available()` 返回 True。

### 构建时找不到 GEOS

确保已执行 `external/geos/geos_with_simd.sh` + `make install`，且 `~/.local/lib/` 下存在 `libgeos.a` 和 `libgeos_c.a`。

### 运行时 `.so` 导入失败

```bash
cd python_planner
python -c "import vec_qmdp_closed_planner"
```
如果报链接错误，检查 `ldd vec_qmdp_closed_planner.so` 输出中是否有 `not found` 的库。

---

## 架构参考

详细的模块关系图、数据流和算法描述见以下文档（均在 `docs/` 目录下）：

| 文档 | 内容 |
|------|------|
| `docs/architecture.md` | 完整模块布局、依赖图、文件对照表 |
| `docs/vec_qmdp.md` | 搜索引擎设计：Static vs Dynamic、UCB 算法、SIMD 回传 |
| `docs/tutorial.md` | 自定义 POMDP 领域实现教程（RockSample 为例） |
| `docs/context_qmdp.md` | 前向仿真引擎：IDM + Stanley + 碰撞检测 |
| `docs/qmdp_trajectory_planner.md` | 顶层规划器接口与流程编排 |
| `docs/trajectory_optimization.md` | 轨迹生成 + LQR 跟踪 + 跨场景评估 |
| `docs/net_belief.md` | 神经网络 belief 管理 + 三阶段 agent 过滤 |
| `docs/reward_function.md` | SIMD 模板化奖励/惩罚函数设计 |
| `docs/state.md` | Ego/Exo 状态表示 + Frenet 转换 + 批量 STRtree 构建 |
| `docs/STRtree.md` | SIMD 加速 R-tree 空间索引设计 |
| `docs/utils.md` | 基础设施：参数常量、SIMD 类型、线程池、内存对齐 |
