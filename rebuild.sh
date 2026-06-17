#!/bin/bash
# 安全重编译脚本：删除旧 .so → 编译 → 验证
set -e

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
OLD_SO=$(find "$SITE_PKG" -path "*/bh_ops/_C*.so" 2>/dev/null || true)

echo "=== 1. 删除旧 .so ==="
if [ -n "$OLD_SO" ]; then
    echo "删除: $OLD_SO"
    rm -f "$OLD_SO"
else
    echo "未找到旧 .so"
fi

echo "=== 2. 清理 build 目录 ==="
rm -rf build/

echo "=== 3. 编译 ==="
TORCH_CUDA_ARCH_LIST="9.0" pip install . --no-build-isolation --force-reinstall 2>&1 | tee /tmp/build_log.txt

echo "=== 4. 检查编译是否成功 ==="
if grep -qi "error" /tmp/build_log.txt; then
    echo "!!! 编译有错误，检查 /tmp/build_log.txt !!!"
    exit 1
fi

NEW_SO=$(find "$SITE_PKG" -path "*/bh_ops/_C*.so" 2>/dev/null || true)
if [ -z "$NEW_SO" ]; then
    echo "!!! 编译后未找到 .so，编译可能静默失败 !!!"
    exit 1
fi

echo "=== 5. 验证 .so 时间戳 ==="
ls -la "$NEW_SO"

echo "=== 6. 验证 import ==="
python -c "import torch; from bh_ops._C import scan_then_fan; print('Import OK')"

echo "=== 完成 ==="
