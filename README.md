# PP-GCN Tennessee Eastman 故障诊断

这是一个精简后的开源工程版本，只保留 Rieth Tennessee Eastman Process 数据集上的主实验和单样本诊断入口。

## 1. 安装依赖

建议使用独立 Python 环境。CPU 可以运行，GPU 会自动使用 CUDA 版 PyTorch。

```powershell
pip install -r requirements.txt
```

如果需要指定 CUDA 版本，请按 PyTorch 官网命令先安装 `torch`，再安装本文件中的其他依赖。

## 2. 下载数据集

主程序使用 Rieth 等人发布在 Harvard Dataverse 的公开 TEP 仿真数据：

- 数据集：Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation
- DOI：https://doi.org/10.7910/DVN/6C3JR1
- 当前主程序需要两个文件：
  - `TEP_FaultFree_Training.RData`
  - `TEP_Faulty_Training.RData`

下载后放到：

```text
data/rieth/
```

最终目录应类似：

```text
data/rieth/TEP_FaultFree_Training.RData
data/rieth/TEP_Faulty_Training.RData
```

注意：`data/` 已在 `.gitignore` 中，不应提交到 Git 仓库。

## 3. 运行主程序

完整运行默认会训练 PCA-SVM、SVM、RandomForest、CNN、PP-GCN 和 Path-GAT：

```powershell
python run_rieth_experiment.py --data-dir data/rieth --output-dir outputs_ppgcn_rieth
```

常用快速测试命令：

```powershell
python run_rieth_experiment.py --epochs 1 --skip-svm --skip-pca --skip-rf --output-dir outputs_smoke
```

运行完成后，输出目录中会包含：

- `metrics.json`：各模型测试指标
- `*_confusion.png`：混淆矩阵
- `best_ppgcn.pt`：PP-GCN 诊断 checkpoint
- `best_path_gat.pt`：Path-GAT 诊断 checkpoint
- `ppgcn_node_contributions.*`：PP-GCN 关键变量解释
- `path_gat_edge_attention.*`：Path-GAT 关键边注意力解释

## 4. 单样本诊断

可以从主程序生成的 `.npz` 缓存中抽取一个测试样本进行诊断：

```powershell
$cache = Get-ChildItem data\rieth -Filter *.npz | Select-Object -First 1 -ExpandProperty FullName
python diagnose.py `
  --checkpoint outputs_ppgcn_rieth\best_path_gat.pt `
  --cache $cache `
  --split test `
  --sample-index 0 `
  --top-n 3 `
  --evidence-top-k 5 `
  --output outputs_ppgcn_rieth\diagnosis_sample0.json
```

也可以传入自己的单窗口文件：

```powershell
python diagnose.py --checkpoint outputs_ppgcn_rieth\best_path_gat.pt --input one_window.npy
```

输入窗口形状应为 `[time, nodes]`，默认主实验是 `[20, 52]`。
