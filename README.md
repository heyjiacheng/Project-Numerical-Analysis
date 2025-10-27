# Project: Numerical Analysis

## 1. Install [Pixi](https://pixi.sh/latest/installation/)

## 2. Clone and Install

```bash
git clone https://github.com/heyjiacheng/Project-Numerical-Analysis.git
cd Project-Numerical-Analysis
pixi install
```

### 3. Start
start environment
```bash
pixi shell
```
run basis function
```bash
python scripts/my_plot_fun.py
```

run static
```bash
python scripts/static_beam.py
```

### 分工
点点要做的
```bash
1. 考虑转角，比较运行结果；
2. 显性组装extended matrix从而求解；
3. 创建类对象时传入材料参数
```
阿澄要做的
```bash
1. 添加bending moment only at end point （现在只有constant load density）。
```