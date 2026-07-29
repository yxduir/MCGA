#!/bin/bash
# 下载 yxdu/MCGA 数据集文件并解压缩
set -e

FILES=("MCGA_train.tar.gz" "MCGA_val.tar.gz" "MCGA_test.tar.gz")

echo "=== 下载数据集文件 ==="
hf download yxdu/MCGA "${FILES[@]}" --repo-type dataset --local-dir .

echo ""
echo "=== 解压缩文件 ==="
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "解压 $f ..."
        tar -zxvf "$f"
        echo "完成: $f"
    else
        echo "警告: $f 未找到，跳过"
    fi
done

echo ""
echo "=== 全部完成 ==="
