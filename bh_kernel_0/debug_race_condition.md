# Prefix Sum Kernel 竞态条件 Debug 记录

## 背景

Scan-then-Fan 三阶段前缀和 kernel，在小输入（n < 1M）下正确，大输入（n >= 2M）随机性失败。

---

## Step 1: 发现问题 — 非确定性失败

小输入稳定通过，大输入随机失败，说明是**竞态条件**而非逻辑错误。

```python
import torch, bh_ops

# 小输入: 100次全通过
n = 512000
for trial in range(100):
    x = torch.randn(n, device='cuda')
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    err = torch.max(torch.abs(out - ref)).item()
    if err > 0.1:
        print(f'FAIL trial {trial}: max_err={err:.4e}')
        break
else:
    print(f'n={n}: 100/100 passed')

# 大输入: 随机失败
n = 4000000
fails = 0
for trial in range(50):
    x = torch.randn(n, device='cuda')
    out = bh_ops.scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    err = torch.max(torch.abs(out - ref)).item()
    if err > 1.0:
        fails += 1
print(f'n={n}: {fails}/50 failures')
```

二分法定位到 n ≈ 1.2M 开始出现偶发失败，n >= 2M 失败率超过 50%。

> **判断依据**: 同一输入多次运行结果不一致 = 非确定性错误 = 竞态条件，而非逻辑错误（逻辑错误每次都会错）。

---

## Step 2: 逐阶段定位

在 .cu 中添加 debug 接口，分别返回每个阶段的中间结果：

```cpp
// 阶段1: 只运行 pre_block_prefix_sum, 返回 blocks 数组
torch::Tensor scan_then_fan_debug(torch::Tensor input) {
    ...
    pre_block_prefix_sum<<<grid_dim,block_dim>>>(...);
    return blocks;
}

// 阶段1: 只运行 pre_block_prefix_sum, 返回 output
torch::Tensor scan_then_fan_stage1_output(torch::Tensor input) {
    ...
    pre_block_prefix_sum<<<grid_dim,block_dim>>>(...);
    return output;
}

// 阶段2: 只运行 prefix_sum_blocks
torch::Tensor scan_then_fan_stage2(torch::Tensor blocks) {
    ...
    prefix_sum_blocks<<<1,kThreadNum>>>(...);
    return blocks;
}
```

```python
from _C import scan_then_fan, scan_then_fan_debug, scan_then_fan_stage2

n = 4000000
perBlockNum = 1024
block_nums = (n + perBlockNum - 1) // perBlockNum

for trial in range(50):
    x = torch.randn(n, device='cuda')

    # Stage1: blocks (每个data block的sum)
    blocks_cuda = scan_then_fan_debug(x)
    blocks_ref = torch.zeros(block_nums, device='cuda')
    for bid in range(block_nums):
        s = bid * perBlockNum
        e = min(s + perBlockNum, n)
        blocks_ref[bid] = x[s:e].sum()
    diff1 = torch.max(torch.abs(blocks_cuda - blocks_ref)).item()

    # Stage2: blocks前缀和
    blocks_cuda2 = scan_then_fan_stage2(blocks_cuda.clone())
    blocks_ref2 = torch.cumsum(blocks_ref, dim=0)
    diff2 = torch.max(torch.abs(blocks_cuda2 - blocks_ref2)).item()

    # 完整结果
    out = scan_then_fan(x)
    ref = torch.cumsum(x, dim=0)
    diff_all = torch.max(torch.abs(out - ref)).item()

    if diff_all > 1.0:
        print(f'Trial {trial} FAIL: full={diff_all:.4e} stage1={diff1:.4e} stage2={diff2:.4e}')
        break
```

**结果**: Stage1 和 Stage2 单独看都是正确的，但完整 pipeline 出错。说明 Stage1 的 **output**（不只是 blocks）有竞态。

> **关键发现**: 阶段1的 blocks（partial sum）是对的，但阶段1写出的 output 有问题。这意味着 `pre_block_prefix_sum` kernel 内部写 output 的逻辑有竞态。

---

## Step 3: compute-sanitizer racecheck 精确定位

### 3.1 简要输出

```bash
/usr/local/cuda/bin/compute-sanitizer --tool racecheck python -c "
import sys, torch
sys.path.insert(0, 'bh_ops_py')
from _C import scan_then_fan
n = 1024*100
x = torch.randn(n, device='cuda')
out = scan_then_fan(x)
"
```

输出：

```
========= Error: Race reported between Read access at pre_block_prefix_sum+0x860
=========     and Write access at pre_block_prefix_sum+0x9d0 [400 hazards]

========= Error: Race reported between Read access at pre_block_prefix_sum+0xea0
=========     and Write access at pre_block_prefix_sum+0xfa0 [412 hazards]

========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

简要模式下只有指令偏移，**没有变量名和共享内存地址**，无法直接定位变量。

### 3.2 详细输出（`--racecheck-report all`）

```bash
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --racecheck-report all python -c "..."
```

输出：

```
========= Error: Potential WAR hazard detected at __shared__ 0x410 in block (11,0,0) :
=========     Read Thread (127,0,0) at pre_block_prefix_sum+0x860
=========     Write Thread (0,0,0) at pre_block_prefix_sum+0x9d0
=========     Current Value : 0, Incoming Value : 181

========= Error: Potential WAR hazard detected at __shared__ 0x411 in block (11,0,0) :
=========     Read Thread (127,0,0) at pre_block_prefix_sum+0x860
=========     Write Thread (0,0,0) at pre_block_prefix_sum+0x9d0

========= Error: Potential WAR hazard detected at __shared__ 0x412 in block (11,0,0) :
=========     Read Thread (127,0,0) at pre_block_prefix_sum+0x860
=========     Write Thread (0,0,0) at pre_block_prefix_sum+0x9d0

========= Error: Potential WAR hazard detected at __shared__ 0x413 in block (11,0,0) :
=========     Read Thread (127,0,0) at pre_block_prefix_sum+0x860
=========     Write Thread (0,0,0) at pre_block_prefix_sum+0x9d0
```

### 3.3 如何从输出推断出竞态变量是 `pre_acc`

racecheck 不直接输出变量名，需要通过以下线索**交叉验证**：

#### 线索 1: 地址跨度 → 变量大小

hazard 跨越的地址：

```
0x410, 0x411, 0x412, 0x413 → 连续 4 字节 = 1 个 float
```

kernel 中有两个 `__shared__` 变量：

```cpp
__shared__ float pre_acc;           // 1 float = 4 字节 ✓ 匹配
__shared__ float block_smem[128];   // 128 float = 512 字节 ✗ 不匹配
```

4 字节的 hazard 只能对应 `pre_acc`，不可能是 `block_smem`。

#### 线索 2: 线程模式 → 读写模式

```
Read  Thread (127,0,0)   → 最后一个线程在读
Write Thread (0,0,0)     → thread 0 在写
```

代码中只有 `pre_acc` 匹配这个模式：

```cpp
// 所有线程读 pre_acc → Read Thread 0~127
output[base+j] = vec[j] + bias + pre_acc;

// 只有 thread 0 写 pre_acc → Write Thread 0
if(tid==0) pre_acc += block_smem[kThreadNum-1];
```

`block_smem` 的模式是每个线程读写自己的槽位 `block_smem[tid]`，不会出现 "Thread 0 写、Thread 127 读同一个位置"。

#### 线索 3: 值的特征 → 变量的语义

```
Current Value : 0, Incoming Value : 181
```

- `Current Value: 0` → 第一轮迭代 `pre_acc` 被初始化为 0
- `Incoming Value: 181` → `pre_acc += block_smem[kThreadNum-1]` 累加了新值

`block_smem` 不存在 "从 0 累加" 的模式。

#### 线索 4: WAR 类型 → 读写顺序

```
Potential WAR hazard → Write After Read
```

即 "先读后写" 竞态：线程 127 还在读，线程 0 已经在写新值。这与代码逻辑完全吻合：

```cpp
output[base+j] = vec[j] + bias + pre_acc;  // ← 先读 (所有线程)
...
if(tid==0) pre_acc += block_smem[kThreadNum-1];  // ← 后写 (thread 0)
// 两者之间没有 __syncthreads()！
```

> **三条线索（地址跨度 4B = 1 float、Thread 0 写其他线程读、值从 0 累加）都指向 `pre_acc`，所以可以确认第一条 race 的变量就是 `pre_acc`。**

### 3.4 第二条 race 如何推断

第二条 race 的输出较少：

```
Race reported between Read at pre_block_prefix_sum+0xea0
    and Write at pre_block_prefix_sum+0xfa0 [412 hazards]
```

没有地址信息，单看无法确定变量。但可以通过**排除法**推断：

1. kernel 中 `__shared__` 变量只有 `pre_acc` 和 `block_smem`
2. `pre_acc` 已被第一条 race 确认
3. 剩下的是 `block_smem` 以及 `block_prefix_sum` 内部声明的 `__shared__ float block_smem_1[4]`
4. 412 个 hazards 的数量级匹配 `block_smem[128]` 的规模

> **第二条 race 更多靠排除法 + 对代码结构的理解推断。最终验证方式：把 `block_prefix_sum` 内联展开、`__shared__` 提到顶层后，racecheck 报 0 hazards，证明推断正确。**

---

## Step 4: 分析竞态根因

### Bug 1: `pre_acc` 读写竞态

原始代码：

```cpp
// 所有线程读 pre_acc (shared memory)
output[base+j] = vec[j] + bias + pre_acc;

// thread 0 写 pre_acc (shared memory)
if(tid==0){
    pre_acc += block_smem[kThreadNum-1];
}
__syncthreads();
```

问题：所有线程读 `pre_acc` 和 thread 0 写 `pre_acc` 之间**没有 `__syncthreads()` 隔离**。

```
warp 0 (含 thread 0):
  读 pre_acc → 写 output → 写 pre_acc(新值) → 进入下一轮迭代

warp 3:
  读 pre_acc → ... → 还在写 output → 读到被覆盖的新 pre_acc！
```

### Bug 2: `block_prefix_sum` 内部 `__shared__` 声明冲突

原始代码 `block_prefix_sum` 是 `__device__` 函数，内部声明了：

```cpp
__device__ void block_prefix_sum(float* smem){
    ...
    __shared__ float block_smem_1[kThreadNum/kWarpSize]; // 内部声明
    ...
}
```

编译器 inline 后，`block_smem_1` 与外部 kernel 的 `pre_acc`、`block_smem` 可能共享同一块 shared memory 区域，导致 `block_prefix_sum` 内部的写操作与外部对 `pre_acc` 的读操作产生不可预测的冲突。

---

## Step 5: 修复

### 修复 1: `pre_acc` 读到寄存器 + 加屏障

```cpp
// 修复: 先读到寄存器, 加 __syncthreads() 隔离读写
float cur_pre_acc = pre_acc;  // 读到寄存器
__syncthreads();              // 确保所有线程读完 pre_acc 后, thread 0 才更新
for(int j = 0; j < kElemst; j++){
    if (base + j < n)
        output[base+j] = vec[j] + bias + cur_pre_acc;
}
if(tid==0){
    pre_acc += block_smem[kThreadNum-1];
}
__syncthreads();
```

### 修复 2: 消除 `__device__` 函数内的 `__shared__` 声明

将 `block_prefix_sum` 内联展开到 `pre_block_prefix_sum` 中，所有 `__shared__` 变量统一声明在 kernel 顶层：

```cpp
__global__ void pre_block_prefix_sum(...) {
    __shared__ float pre_acc;
    __shared__ float block_smem[kThreadNum];
    __shared__ float warp_smem[kThreadNum/kWarpSize]; // 从 block_prefix_sum 移出来
    ...
    // inline block_prefix_sum 的逻辑, 直接操作上面的 shared 变量
}
```

### 修复 3: `add_bias_kernel` 移除 shared memory

```cpp
// 原始: 用 shared memory block_bias + grid-stride loop
// 问题: 循环末尾缺 __syncthreads(), thread 0 写新 block_bias 与其他线程读旧值竞争
// 修复: 每个线程直接从 global memory 读 bias, blocks[block_id-1] 已在 L2 cache,
//       128 个线程读同一地址会被硬件合并, 性能损失极小, 彻底消除 shared memory 竞态
__global__ void add_bias_kernel(float* output,float* blocks,int block_nums,int n){
    for(int block_id=blockIdx.x;block_id<block_nums;block_id+=gridDim.x){
        if (block_id==0) continue;
        float bias = blocks[block_id-1];  // 直接读 global memory
        for(int i=threadIdx.x; i<perBlockNum/kElemst; i+=kThreadNum){
            int base = block_id*perBlockNum+i*kElemst;
            for(int j=0;j<kElemst;j++){
                if(base+j<n) output[base+j] += bias;
            }
        }
    }
}
```

---

## Step 6: 验证

### 6.1 racecheck 清零

```bash
/usr/local/cuda/bin/compute-sanitizer --tool racecheck python -c "
import sys, torch
sys.path.insert(0, 'bh_ops_py')
from _C import scan_then_fan
n = 4_000_000
x = torch.randn(n, device='cuda')
out = scan_then_fan(x)
"
```

```
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

### 6.2 大规模稳定性测试

```python
for n in [4_000_000, 10_000_000, 50_000_000]:
    fails = 0
    for trial in range(50):
        x = torch.randn(n, device='cuda')
        out = scan_then_fan(x)
        ref = torch.cumsum(x, dim=0)
        err = torch.max(torch.abs(out - ref)).item()
        if err > 1.0:
            fails += 1
    print(f'n={n}: {fails}/50 failures')
```

结果：

```
n=4000000: 0/50 failures
n=10000000: 0/50 failures
n=50000000: 0/50 failures
```

---

## 关键经验

1. **非确定性失败 = 竞态条件**：如果同一输入多次运行结果不一致，一定是竞态
2. **`__syncthreads()` 是唯一同步手段**：for 循环、if 分支都没有同步语义
3. **`__shared__` 不要在 `__device__` 函数内声明**：inline 后 shared memory 布局不可控，可能导致跨变量竞争
4. **shared memory 读写必须加屏障**：特别是"多线程读 + 单线程写"模式，读之前先拷贝到寄存器，再 `__syncthreads()`，然后再写
5. **`compute-sanitizer --tool racecheck` 是定位竞态的利器**：简要模式只给指令偏移，`--racecheck-report all` 才给 `__shared__` 地址和线程信息
6. **racecheck 输出不会直接告诉你变量名**：需要通过地址跨度（推断变量大小）、线程模式（推断读写者）、值特征（推断变量语义）三条线索交叉验证

## 附录: racecheck 输出解读速查

| 输出字段 | 含义 | 如何用于定位 |
|---------|------|-------------|
| `__shared__ 0x410~0x413` | 竞态的 shared memory 地址范围 | 范围 = 4B → 1 float → 对应 `pre_acc` |
| `Read Thread (127,0,0)` | 读操作的线程 ID | Thread 127 读 → "所有线程读" 模式 |
| `Write Thread (0,0,0)` | 写操作的线程 ID | Thread 0 写 → "单线程写" 模式 |
| `Current Value: 0` | 读到的旧值 | 初始化为 0 → 匹配 `pre_acc = 0.0f` |
| `Incoming Value: 181` | 将要写入的新值 | 累加操作 → 匹配 `pre_acc += ...` |
| `[400 hazards]` | 同类竞态的触发次数 | 数量大 = 高频共享变量 |
| `WAR` (Write After Read) | 先读后写竞态 | 读和写之间缺 `__syncthreads()` |
