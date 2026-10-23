"""
설치된 Pretendard 가변 글꼴 → 정적 글꼴 4개 (Regular·Medium·SemiBold·Bold)
PowerPoint 는 가변 글꼴을 PDF 에 글꼴로 넣지 못해(윤곽선으로 변환) 정적 인스턴스를 만들어 씀.
실행: python fonts.py <출력 폴더>   (export.ps1 이 호출, 설치하지 않고 세션에만 불러옴)
"""
import os
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

SRC = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts" / "PretendardVariable.ttf"
# (굵기, 글꼴 이름, 하위 이름, 스타일)
SPECS = [(400, "Pretendard", "Regular", "Regular"), (700, "Pretendard", "Bold", "Bold"),
         (500, "Pretendard Medium", "Regular", "Medium"), (600, "Pretendard SemiBold", "Regular", "SemiBold")]


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for wt, fam, sub, style in SPECS:
        out = out_dir / f"Pretendard-{style}.ttf"
        if out.exists():
            continue
        f = instantiateVariableFont(TTFont(SRC), {"wght": wt})
        n = f["name"]
        for nid in (1, 2, 3, 4, 6, 16, 17, 21, 22, 25):
            n.removeNames(nameID=nid)
        ps = "Pretendard-" + style
        for nid, val in [(1, fam), (2, sub), (3, ps + ";static"), (4, "Pretendard " + style), (6, ps),
                         (16, "Pretendard"), (17, style)]:
            n.setName(val, nid, 3, 1, 0x409)
            n.setName(val, nid, 1, 0, 0)
        bold = sub == "Bold"
        os2 = f["OS/2"]
        os2.usWeightClass = wt
        os2.fsSelection = (os2.fsSelection & ~0b1100001) | (0b100000 if bold else 0b1000000)
        f["head"].macStyle = 1 if bold else 0
        f.save(out)
        print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
