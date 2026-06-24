from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK1 = ROOT / "task1" / "report" / "main.pdf"
TASK2 = ROOT / "task2" / "report" / "main.pdf"
APPENDIX = OUT_DIR / "submission_links.pdf"
COMBINED = OUT_DIR / "HW3_深度学习与空间智能_综合提交.pdf"
TASK1_COPY = ROOT / "task1" / "report" / "combined_submission.pdf"


def register_font() -> tuple[str, str, str]:
    simsun = Path(r"C:\Windows\Fonts\simsun.ttc")
    simhei = Path(r"C:\Windows\Fonts\simhei.ttf")
    times = Path(r"C:\Windows\Fonts\times.ttf")
    cn = "Helvetica"
    cn_bold = "Helvetica-Bold"
    en = "Times-Roman"
    if simsun.exists():
        pdfmetrics.registerFont(TTFont("SimSun", str(simsun)))
        cn = "SimSun"
    if simhei.exists():
        pdfmetrics.registerFont(TTFont("SimHei", str(simhei)))
        cn_bold = "SimHei"
    if times.exists():
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(times)))
        en = "TimesNewRoman"
    return cn, cn_bold, en


CN, CN_BOLD, EN = register_font()


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def build_appendix() -> None:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=CN_BOLD,
            fontSize=18,
            leading=25,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=CN_BOLD,
            fontSize=14,
            leading=21,
            wordWrap="CJK",
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=CN,
            fontSize=12,
            leading=19,
            firstLineIndent=24,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=7,
        ),
        "mono": ParagraphStyle(
            "mono",
            parent=base["Code"],
            fontName=CN,
            fontSize=9.2,
            leading=13,
            leftIndent=12,
            rightIndent=12,
            firstLineIndent=0,
            backColor=colors.HexColor("#f5f5f5"),
            borderColor=colors.HexColor("#d8d8d8"),
            borderWidth=0.45,
            borderPadding=6,
            wordWrap="CJK",
            spaceBefore=4,
            spaceAfter=8,
        ),
    }

    def p(text: str, style: str = "body") -> Paragraph:
        return Paragraph(esc(text), styles[style])

    story = [
        Spacer(1, 0.8 * cm),
        HRFlowable(width="100%", thickness=4.0, color=colors.black, spaceBefore=0, spaceAfter=22),
        p("提交链接与权重文件说明", "title"),
        HRFlowable(width="100%", thickness=1.0, color=colors.black, spaceBefore=2, spaceAfter=18),
        p("本页汇总代码仓库、报告位置和需要单独上传的权重/结果文件。GitHub 仓库只保存代码、报告、可视化图和环境说明；大型数据集、预训练权重、PLY/GLB 和视频文件单独上传。"),
        p("代码仓库", "h1"),
        p("https://github.com/chen12138-123/HW3--", "mono"),
        p("报告文件", "h1"),
        p("task1/report/main.pdf\ntask2/report/main.pdf\nreport/HW3_combined_submission.pdf", "mono"),
        p("权重与结果文件链接", "h1"),
        p(
            "Task1 权重/结果链接：待上传至云盘或课程平台后填写。本地需要打包的文件如下。",
            "mono",
        ),
        p(
            "task1/output/real_3dgs_bg/point_cloud/iteration_7000/point_cloud.ply; "
            "task1/output/object_a_official_3dgs/point_cloud/iteration_7000/point_cloud.ply; "
            "task1/output/final_assets/object_b_final/mesh.glb; "
            "task1/output/final_assets/object_b_final/model.ply; "
            "task1/output/final_assets/object_c_final/mesh.glb; "
            "task1/output/final_assets/object_c_final/model.ply; "
            "task1/output/final_fused/model.ply; "
            "task1/output/final_fused/roaming_video.mp4; "
            "task1/data/models/TripoSR/model.ckpt（如需复现实验中的 B/C 资产生成）。",
            "mono",
        ),
        p("Task2 权重链接：https://pan.baidu.com/s/1voGnHTbvz8gjdoWDuJbzug?pwd=mjxn", "mono"),
        p(
            "该链接应包含或补充以下 checkpoint：task2/checkpoints/A_only_2000/pretrained_model; "
            "task2/checkpoints/ABC_joint_2000/pretrained_model; "
            "task2/checkpoints/A_only_10000/pretrained_model; "
            "task2/checkpoints/ABC_joint_10000/pretrained_model。"
            "这四个 checkpoint 需要从训练 task2 的机器或云端输出目录上传。",
            "mono",
        ),
        p("组员信息", "h1"),
        p("周湘洋、毛琦骏、陈希"),
    ]

    doc = SimpleDocTemplate(
        str(APPENDIX),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.35 * cm,
        title="HW3 submission links",
        author="周湘洋、毛琦骏、陈希",
    )
    doc.build(story)


def merge_pdfs() -> None:
    writer = PdfWriter()
    for path in [TASK1, TASK2, APPENDIX]:
        reader = PdfReader(str(path))
        for page in reader.pages:
            writer.add_page(page)
    with COMBINED.open("wb") as f:
        writer.write(f)
    TASK1_COPY.parent.mkdir(parents=True, exist_ok=True)
    TASK1_COPY.write_bytes(COMBINED.read_bytes())


def main() -> None:
    build_appendix()
    merge_pdfs()
    print(COMBINED)
    print(TASK1_COPY)


if __name__ == "__main__":
    main()
