# Git 发布前检查清单

在暂存、提交或推送前，按以下顺序完成检查。

1. 检查工作区，确认没有意外文件或不应提交的本地配置。

   ```powershell
   git status --short
   ```

2. 只暂存已审阅的文件，并列出暂存区内容。

   ```powershell
   git add <已审阅文件>
   git diff --cached --name-status
   ```

3. 对暂存差异执行敏感词扫描。发现命中时先移除或替换敏感内容，再重新检查。

   ```powershell
   git diff --cached --text | rg -n -i "api[_-]?key|secret|token|password|credential|private[_-]?key"
   ```

4. 检查暂存文件大小；模型、原始数据、日志或其他大文件不应进入提交。

   ```powershell
   git diff --cached --name-only | ForEach-Object { Get-Item -LiteralPath $_ } |
     Select-Object FullName, Length | Where-Object Length -gt 10MB
   ```

5. 再次确认忽略规则生效。`request.md`、`.env`、密钥、模型、原始数据和运行日志必须保持未跟踪；`docs/project-log.md` 可以跟踪。

   ```powershell
   git check-ignore -v request.md .env logs/example.log data/raw/example.csv
   git check-ignore -v docs/project-log.md
   ```

6. 复查暂存差异后再提交。

   ```powershell
   git diff --cached --check
   git diff --cached
   ```
