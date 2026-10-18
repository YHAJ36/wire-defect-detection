#!/usr/bin/env bash
# 第三批链的"脱离式启动"入口。
#
# 为什么需要这个文件（2026-10-16 踩坑）
#   直接用 PowerShell `Start-Process -ArgumentList '-lc','cd /d/2 && exec bash ...'`
#   启动，Start-Process 会把数组用空格拼起来、引号丢失，bash 收到的 argv 被拆错，
#   于是 bash **静默退出**（stdout/stderr 都是 0 字节），链根本没起来。
#   把要执行的命令写进**文件**、只把文件路径当唯一参数传，就绕开了全部引号问题。
#
# 用法（唯一参数是脚本路径，不含空格）：
#   powershell -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Git\bin\bash.exe' \
#       -ArgumentList 'D:/2/scripts/_chain3_entry.sh' -WindowStyle Hidden -PassThru"
set -u
cd /d/2 || exit 1
exec bash /d/2/scripts/run_batch3.sh >> /d/2/exp_upscale/logs/chain_batch3_stdout.log 2>&1
