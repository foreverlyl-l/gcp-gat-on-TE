# PP-GCN Tennessee Eastman 故障诊断

## 1. 安装依赖

```powershell
pip install -r requirements.txt
```

## 2. 下载数据集

使用 Rieth 等发布在 Harvard Dataverse 的公开 TEP 仿真数据：

- 数据集：Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation
- DOI：https://doi.org/10.7910/DVN/6C3JR1
- 需要两个文件：
  - `TEP_FaultFree_Training.RData`
  - `TEP_Faulty_Training.RData`

下载后放到：

```text
data/rieth/TEP_FaultFree_Training.RData
data/rieth/TEP_Faulty_Training.RData
```

## 3. 运行主程序

完整运行默认会训练 PCA-SVM、SVM、RandomForest、CNN、PP-GCN 和 Path-GAT做测速：

```powershell
python run_rieth_experiment.py --data-dir data/rieth --output-dir outputs_ppgcn_rieth
```

输出包含：

- `metrics.json`：各模型测试指标
- `*_confusion.png`：混淆矩阵
- `best_ppgcn.pt`：PP-GCN 诊断 checkpoint
- `best_path_gat.pt`：Path-GAT 诊断 checkpoint
- `ppgcn_node_contributions.*`：PP-GCN 关键变量解释
- `path_gat_edge_attention.*`：Path-GAT 关键边注意力解释

## 4. 单样本诊断

可从主程序生成的 `.npz` 缓存中抽取单样本样本进行诊断：

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

或传入自己的单窗口文件：

```powershell
python diagnose.py --checkpoint outputs_ppgcn_rieth\best_path_gat.pt --input one_window.npy
```

## 5. License

MIT License. Copyright (c) 2026 foreverlyl-l.
