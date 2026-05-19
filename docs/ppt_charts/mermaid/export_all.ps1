# Mermaid图表批量导出脚本
# 使用方法: 在PowerShell中运行 .\export_all.ps1

$inputDir = "d:\量化\V2.0\docs\ppt_charts\mermaid"
$outputDir = "d:\量化\V2.0\docs\ppt_charts\mermaid\output"

# 创建输出目录
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
    Write-Host "创建输出目录: $outputDir" -ForegroundColor Green
}

# 获取所有.md文件（排除README.md）
$files = Get-ChildItem -Path $inputDir -Filter "slide*.md"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始批量导出Mermaid图表为PNG" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$successCount = 0
$failCount = 0

foreach ($file in $files) {
    $baseName = $file.BaseName
    $inputFile = $file.FullName
    $outputFile = Join-Path $outputDir "$baseName.png"
    
    Write-Host "正在导出: $($file.Name) -> $baseName.png" -NoNewline
    
    try {
        # 使用mmdc导出，设置白色背景、高分辨率
        mmdc -i "$inputFile" -o "$outputFile" -b white -s 2 2>$null
        
        if (Test-Path $outputFile) {
            Write-Host " [成功]" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host " [失败]" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host " [错误: $_]" -ForegroundColor Red
        $failCount++
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "导出完成!" -ForegroundColor Cyan
Write-Host "成功: $successCount 个" -ForegroundColor Green
Write-Host "失败: $failCount 个" -ForegroundColor Red
Write-Host "输出目录: $outputDir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 列出所有生成的文件
Write-Host "`n生成的文件列表:" -ForegroundColor Yellow
Get-ChildItem -Path $outputDir -Filter "*.png" | ForEach-Object {
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "  - $($_.Name) (${size} KB)" -ForegroundColor White
}
