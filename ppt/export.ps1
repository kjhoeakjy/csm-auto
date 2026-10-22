# PowerPoint 로 글꼴 포함 저장 → PDF(글꼴 포함)와 미리보기 이미지 내보내기
# 정적 Pretendard 글꼴을 만들어 이번 Windows 세션에만 불러옴 (설치·레지스트리 변경 없음, 재부팅하면 사라짐)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Split-Path -Parent (Split-Path -Parent $root)
$src  = Join-Path $proj "ppt\src"
$pptx = Join-Path $root "역량PT_김연주.pptx"
$pdf  = Join-Path $root "역량PT_김연주.pdf"
$fd   = Join-Path $env:TEMP "pretendard_static"
& (Join-Path $proj ".venv\Scripts\python.exe") (Join-Path $src "fonts.py") $fd
Add-Type -Namespace W -Name Gdi -MemberDefinition @'
[DllImport("gdi32.dll", CharSet = CharSet.Unicode)] public static extern int AddFontResource(string f);
[DllImport("user32.dll")] public static extern int SendMessageTimeout(IntPtr h, uint m, IntPtr w, IntPtr l, uint f, uint t, out IntPtr r);
'@
Get-ChildItem $fd -Filter *.ttf | ForEach-Object { [void][W.Gdi]::AddFontResource($_.FullName) }
$r = [IntPtr]::Zero; [void][W.Gdi]::SendMessageTimeout([IntPtr]0xffff, 0x1D, [IntPtr]::Zero, [IntPtr]::Zero, 2, 1000, [ref]$r)

$pp = New-Object -ComObject PowerPoint.Application
$d = $pp.Presentations.Open($pptx, $false, $false, $false)
$tmp = Join-Path $env:TEMP "역량PT_embed.pptx"
$d.SaveAs($tmp, 24, -1)                   # 24 = pptx, -1 = TrueType 글꼴 포함 (같은 경로 덮어쓰기는 동기화 잠금으로 실패 → 임시 저장 후 복사)
$d.SaveAs($pdf, 32)                       # 32 = PDF (사용 글꼴 포함)
foreach ($s in $d.Slides) { $s.Export((Join-Path $root ("preview_{0}.png" -f $s.SlideIndex)), "PNG", 1920, 1080) }
$d.Close(); $pp.Quit()
Copy-Item $tmp $pptx -Force
Get-ChildItem (Join-Path $root "역량PT_김연주.*"), (Join-Path $root "preview_*.png") | Select-Object Name, Length


