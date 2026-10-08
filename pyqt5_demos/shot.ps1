param([Parameter(Mandatory=$true)][string]$Out)

# 截取 python 进程中面积最大的可见窗口。
# 用 PrintWindow(PW_RENDERFULLCONTENT) 直接抓窗口自身内容，
# 不依赖窗口是否在最前 —— 避免截到盖在上面的其它窗口。

Add-Type -AssemblyName System.Drawing

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class Win {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
}
"@

$script:cands = New-Object System.Collections.ArrayList

$cb = [Win+EnumWindowsProc]{
    param($h, $l)
    if ([Win]::IsWindowVisible($h)) {
        $len = [Win]::GetWindowTextLength($h)
        if ($len -gt 0) {
            $sb = New-Object System.Text.StringBuilder ($len + 1)
            [void][Win]::GetWindowText($h, $sb, $sb.Capacity)
            $title = $sb.ToString()
            $procId = [uint32]0
            [void][Win]::GetWindowThreadProcessId($h, [ref]$procId)
            $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($p -and $p.ProcessName -like 'python*') {
                $r = New-Object Win+RECT
                [void][Win]::GetWindowRect($h, [ref]$r)
                $w = $r.Right - $r.Left
                $ht = $r.Bottom - $r.Top
                if ($w -gt 300 -and $ht -gt 200) {
                    [void]$script:cands.Add([PSCustomObject]@{
                        Handle = $h; Title = $title; W = $w; H = $ht; Area = $w * $ht
                    })
                }
            }
        }
    }
    return $true
}

[void][Win]::EnumWindows($cb, [IntPtr]::Zero)

if ($script:cands.Count -eq 0) {
    Write-Output 'NO_WINDOW'
    exit 1
}

$best = $script:cands | Sort-Object Area -Descending | Select-Object -First 1

$bmp = New-Object System.Drawing.Bitmap $best.W, $best.H
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
$ok = [Win]::PrintWindow($best.Handle, $hdc, 2)   # 2 = PW_RENDERFULLCONTENT
$g.ReleaseHdc($hdc)
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

Write-Output ("PrintWindow=" + $ok + "|" + $best.Title + "|" + $best.W + "x" + $best.H)
