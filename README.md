# MEGA-MOE-TD

基于[Triton-dist-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)项目开发MOE kernel的仓库。

配置运行环境有两种途径：

- 途径一：前往[Triton-dist-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)源仓库进行环境配置。

- 途径二：选取合一的Triton-Ascend包（集成triton-dist）进行开发。

## 仓库结构

```
Mega-MoE-TD/
├── benchmark/
│   └── moe_backward_golden.py
├── kernels/
│   ├── common.py
│   ├── dispatch_fc2_bwd.py
│   ├── swiglu_bwd.py
│   ├── transposed_grouped_gemm.py
│   └── combine_fc1_bwd.py
├── functions/
│   └── moe_backward.py
└── test/
    ├── layer/run_moe_backward.py
    └── function/test_moe_backward_function.py
```

## 运行


```bash
# golden 后向 vs autograd 交叉验证
torchrun --nproc-per-node=2  benchmark/moe_backward_golden.py

# 端到端 triton vs torch 精度 + 性能
torchrun --nproc-per-node=2 test/layer/run_moe_backward.py
```

## 现有性能结果

