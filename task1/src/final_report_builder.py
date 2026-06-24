from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)


def register_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("CN", str(path)))
            return "CN"
    return "Helvetica"


FONT = register_font()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#203040"),
            wordWrap="CJK",
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT,
            fontSize=12.3,
            leading=17,
            textColor=colors.HexColor("#203040"),
            wordWrap="CJK",
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.3,
            leading=14.4,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.8,
            leading=11,
            wordWrap="CJK",
            textColor=colors.HexColor("#404040"),
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.HexColor("#505050"),
            spaceAfter=6,
        ),
    }


S = make_styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), S[style])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item, "body"), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def img(name: str, width: float, caption: str | None = None, max_height: float | None = None) -> list:
    path = FIG / name
    if not path.exists():
        raise FileNotFoundError(path)
    with PILImage.open(path) as im:
        w, h = im.size
    height = width * h / w
    max_height = max_height or 18.2 * cm
    if height > max_height:
        scale = max_height / height
        width *= scale
        height *= scale
    flowables: list = [Image(str(path), width=width, height=height)]
    if caption:
        flowables.append(P(caption, "caption"))
    return flowables


def table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    data = [[P(cell, "small") for cell in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D6DC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#203040")),
        ]
    t.setStyle(TableStyle(style))
    return t


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 0.75 * cm, f"HW3 Task 1 Report - Page {doc.page}")
    canvas.restoreState()


def story() -> list:
    page_w = A4[0] - 3.2 * cm
    elems: list = []

    elems.append(P("题目一：基于 3DGS 与 AIGC 的多源 3D 资产生成与真实场景融合", "title"))
    elems.append(P("最终修订报告：使用真实 3DGS / threestudio / Zero123 渲染输出，剔除旧点云占位图", "caption"))
    elems.append(P("项目目录：C:\\Desktop\\HW3-深度学习空间智能。生成日期：2026-06-23。提交前请补充姓名、学号与分工。", "small"))
    elems.extend(img("pipeline_repro_routes.png", page_w, "图 1 复现路线总览：背景和 A 采用 GraphDECO 3DGS；B 采用 threestudio DreamFusion/SDS；C 采用 Zero123 单图到 3D。"))

    elems.append(P("1. 任务要求与修订原则", "h1"))
    elems.append(P("题目一要求完成一个统一场景中的多源 3D 资产生成任务：先重建一个开源真实背景场景，再通过三条不同技术路线获得三个独立物体资产，最后给出多视角渲染、对比记录和报告。上一版报告最大问题是把点云散点图或示意图放在最终结果位置。本版只使用训练程序实际导出的渲染图，3DGS 部分使用 render/GT 对比，生成式部分使用 threestudio 测试阶段的 RGB / normal / opacity 三联图和多视角 RGB 网格。", "body"))
    elems.append(bullets([
        "背景：从 Mip-NeRF 360 数据集中选择 garden 场景，用 GraphDECO 3D Gaussian Splatting 训练并渲染，作为统一环境背景。",
        "物体 A：真实多视角重建路线。使用官方公开 truck 多视角数据替代本机无法现场采集的手机环绕视频，保留 COLMAP 位姿 + 3DGS 的完整技术路径。",
        "物体 B：文本到 3D 路线。使用 threestudio DreamFusion 配置、Stable Diffusion v1.5 和 SDS loss，从文本 prompt 生成可渲染隐式体。",
        "物体 C：单图到 3D 路线。使用带透明前景的单张图片作为条件输入，结合 Zero123-diffusers 先验优化 3D 隐式体并导出多视角渲染。",
        "所有实验均记录输出目录、日志、训练步数和视觉结果；失败或低质量实验作为对比说明，不再包装成成功。"
    ]))

    elems.append(P("2. 开源算法与运行环境", "h1"))
    elems.append(table([
        ["模块", "开源来源 / 权重", "本次作用"],
        ["3D Gaussian Splatting", "https://github.com/graphdeco-inria/gaussian-splatting", "背景 garden 与物体 A truck 的 3DGS 训练、渲染和 L1/PSNR 评估。"],
        ["Mip-NeRF 360", "https://jonbarron.info/mipnerf360/", "选择 garden 作为统一环境背景，满足题目指定的开源 3D 场景数据要求。"],
        ["Tanks and Temples truck", "GraphDECO 官方 3DGS 数据包", "作为真实多视角物体 A 的替代数据，包含 COLMAP 相机和真实图像。"],
        ["threestudio", "https://github.com/threestudio-project/threestudio", "物体 B 的 DreamFusion/SDS 与物体 C 的 Zero123-simple 复现实验框架。"],
        ["Stable Diffusion v1.5", "runwayml/stable-diffusion-v1-5", "物体 B 文本到 3D 的 2D 扩散先验。SD-2.1-base 访问受限，因此换用公开可下载权重。"],
        ["Zero123 diffusers", "bennyguo/zero123-diffusers", "物体 C 单图到 3D 的视角条件扩散先验。"],
    ], [3.2 * cm, 6.6 * cm, 7.3 * cm]))
    elems.append(P("运行环境为 Windows + Anaconda Python 3.10、PyTorch 2.1.2+cu118、CUDA 11.8、NVIDIA GeForce RTX 4080 Laptop GPU。Windows 下为了编译 nerfacc CUDA 扩展，使用 Visual Studio 2019 vcvars64 环境，并在 nerfacc JIT 编译参数中加入 -allow-unsupported-compiler。该修改只影响扩展编译，不改变算法公式和训练目标。", "body"))

    elems.append(PageBreak())
    elems.append(P("3. 背景场景：Mip-NeRF 360 garden + 3DGS", "h1"))
    elems.append(P("背景严格从题目建议的数据集中选择 Mip-NeRF 360 的 garden 场景。训练使用 GraphDECO 官方 train.py，输入为 garden 的 COLMAP 位姿和 images_4 图像，分辨率参数为 -r 4，开启 eval split，训练 7000 iterations 后使用官方 render.py 导出 train/test 视角。", "body"))
    elems.append(table([
        ["项目", "设置 / 结果"],
        ["数据集", "Mip-NeRF 360 garden, images_4"],
        ["算法", "GraphDECO 3D Gaussian Splatting"],
        ["训练步数", "7000 iterations"],
        ["测试集指标", "L1 = 0.0195519, PSNR = 30.4103 dB"],
        ["训练集指标", "L1 = 0.0149199, PSNR = 33.6060 dB"],
        ["输出目录", "output/real_3dgs_bg"],
    ], [4.1 * cm, 12.7 * cm], header=False))
    elems.extend(img("real_3dgs_garden_render_gt_compare.png", page_w, "图 2 garden 3DGS 的 test render 与 GT 对比。该图来自 output/real_3dgs_bg/test/ours_7000/renders 和 gt。"))
    elems.extend(img("real_3dgs_garden_contact.png", page_w, "图 3 garden 多视角 novel-view rendering 联系表，可见是连续渲染图而非点云散点。"))

    elems.append(PageBreak())
    elems.append(P("4. 物体 A：真实多视角 3DGS 重建", "h1"))
    elems.append(P("物体 A 按题目要求应由手机环绕视频或多视角照片经 COLMAP 和 3DGS 重建。由于当前环境无法补采手机视频，本次使用 GraphDECO 官方公开数据中的 Tanks and Temples truck 作为真实多视角替代。该数据同样包含真实图像、COLMAP 位姿和稀疏点，因此可以完整复现“真实多视角输入 - COLMAP - 3DGS - novel view rendering”的路径。", "body"))
    elems.append(table([
        ["项目", "设置 / 结果"],
        ["数据", "data/official_3dgs/tandt/truck"],
        ["算法", "COLMAP 位姿 + GraphDECO 3D Gaussian Splatting"],
        ["训练步数", "7000 iterations"],
        ["测试集指标", "L1 = 0.0317515, PSNR = 26.1683 dB"],
        ["训练集指标", "L1 = 0.0252217, PSNR = 27.8280 dB"],
        ["输出目录", "output/object_a_official_3dgs"],
    ], [4.1 * cm, 12.7 * cm], header=False))
    elems.extend(img("real_3dgs_object_a_truck_render_gt_compare.png", page_w, "图 4 物体 A truck 的 3DGS render / GT 对比。"))
    elems.extend(img("real_3dgs_object_a_truck_contact.png", page_w, "图 5 物体 A truck 的多视角渲染结果。"))

    elems.append(PageBreak())
    elems.append(P("5. 物体 B：文本到 3D 生成", "h1"))
    elems.append(P("物体 B 使用 threestudio 的 dreamfusion-sd 配置，核心为 Stable Diffusion 文本图像先验和 SDS loss。第一次使用“glossy red toy sports car”作为 prompt 训练到 2000 steps，虽然流程完整并导出了 60 视角测试图，但结果仍是红色团块，说明该复杂细长物体在短训 SDS 中不稳定。为得到可用的文本到 3D 渲染结果，补充实验改用更适合短训收敛的 prompt：“a shiny red apple with a small green leaf, single object, centered, high detail, studio lighting, white background”，并加入 negative prompt、拉远测试相机和更强 sparsity。", "body"))
    elems.append(table([
        ["实验", "Prompt / 设置", "结果判断"],
        ["B-0 失败对比", "glossy red toy sports car, 2000 steps, SD v1.5, SDS", "流程跑通，但形体未成车，作为失败对比记录。"],
        ["B-1 最终采用", "shiny red apple with a small green leaf, 1800 steps, SD v1.5, SDS", "形体可辨认，RGB/normal/opacity 均有稳定物体轮廓。"],
        ["输出日志", "output/logs/threestudio_b_apple_sds_1800.log", "Trainer.fit stopped: max_steps=1800 reached 后导出 60 个 test views。"],
    ], [3.1 * cm, 8.0 * cm, 5.8 * cm]))
    elems.extend(img("threestudio_b_sds_2000_rgb_views_grid.png", page_w, "图 6 B-0 跑车 prompt 的 2000-step 失败对比：可见主要是红色体块，不能作为最终物体。"))
    elems.extend(img("threestudio_b_apple_sds_1800_progress_rgb.png", page_w, "图 7 B-1 apple prompt 的训练过程 RGB 截图：450/900/1350/1800 steps。"))
    elems.extend(img("threestudio_b_apple_sds_1800_rgb_views_grid.png", page_w, "图 8 B-1 apple 的最终多视角 RGB 渲染网格，来自 it1800-test。"))
    elems.extend(img("threestudio_b_apple_sds_1800_triplet_030.png", page_w * 0.82, "图 9 B-1 apple 单帧三联输出：RGB / normal / opacity。"))

    elems.append(PageBreak())
    elems.append(P("6. 物体 C：单图到 3D 生成", "h1"))
    elems.append(P("物体 C 使用单张前景图作为条件输入，并通过 Zero123-diffusers 视角先验进行 3D 优化。最初尝试 futuristic_car_rgba.png 的短训结果仍偏灰块；最终采用主体更居中、轮廓更清晰的 hamburger_rgba.png 作为输入，训练 600 steps 后导出 60 个测试视角。该结果保留了汉堡的红/黄/深色层次，normal 与 opacity 也显示出稳定体积轮廓，适合作为单图到 3D 路线的最终展示。", "body"))
    elems.append(table([
        ["项目", "设置 / 结果"],
        ["输入图", "load/images/hamburger_rgba.png，透明前景单物体"],
        ["算法", "threestudio zero123-simple，bennyguo/zero123-diffusers，SDS"],
        ["训练设置", "max_steps=600, train 64x64, eval/test 128x128, n_test_views=60"],
        ["日志", "output/logs/threestudio_c_zero123_hamburger_600_retry.log"],
        ["输出目录", "outputs/zero123-simple/sds_64_hamburger_rgba.png@20260623-072805"],
    ], [4.1 * cm, 12.7 * cm], header=False))
    elems.extend(img("object_c_input_hamburger_rgba.png", page_w * 0.42, "图 10 C 最终采用的单图输入：hamburger_rgba.png，透明前景单物体。", max_height=7.2 * cm))
    elems.extend(img("threestudio_c_zero123_hamburger_600_progress_rgb.png", page_w, "图 11 C hamburger 的训练过程 RGB 截图：200/400/600 steps。"))
    elems.extend(img("threestudio_c_zero123_hamburger_600_rgb_views_grid.png", page_w, "图 12 C hamburger 的最终多视角 RGB 渲染网格，来自 it600-test。"))
    elems.extend(img("threestudio_c_zero123_hamburger_600_triplet_030.png", page_w * 0.82, "图 13 C hamburger 单帧三联输出：RGB / normal / opacity。"))

    elems.append(PageBreak())
    elems.append(P("7. 统一背景融合与实验对比", "h1"))
    elems.append(P("融合阶段以 garden 3DGS 渲染图作为统一环境背景，将 A/B/C 的代表性渲染结果进行同一画面 compositing，目的是展示三个独立资产在统一场景中的相对位置、尺度和视觉风格。严格来说，A 是 3DGS 渲染资产，B/C 是 threestudio 隐式体渲染资产；本次报告的融合图是 2D 层面的结果汇总，不伪装成已经完成物理一致的三维光照融合。", "body"))
    elems.extend(img("fusion_garden_abc_composite.png", page_w, "图 14 统一 garden 背景上的 A/B/C 资产融合展示。"))
    elems.append(table([
        ["任务", "技术路径", "最终结果质量", "主要问题与改进"],
        ["背景", "Mip-NeRF 360 garden + 3DGS", "PSNR 30.41 dB，渲染清晰", "已满足题目指定真实场景重建要求。"],
        ["物体 A", "真实多视角 truck + COLMAP + 3DGS", "PSNR 26.17 dB，车体结构清楚", "用官方真实多视角数据替代本机手机采集。"],
        ["物体 B", "Text prompt + DreamFusion/SDS", "apple 可辨认，跑车失败作为对照", "SDS 对 prompt 与相机尺度敏感；复杂细长物体需要更长训练或 VSD/MVDream。"],
        ["物体 C", "Single RGBA image + Zero123", "hamburger 颜色和轮廓可辨认", "短训纹理噪声仍明显；原版 Zero123/Stable-Zero123 权重体积较大，受本机显存与下载时间限制。"],
    ], [2.4 * cm, 4.4 * cm, 4.5 * cm, 5.6 * cm]))

    elems.append(P("8. 结论", "h1"))
    elems.append(P("本次修订完成了题目一所需的四条主要链路：真实背景 3DGS、真实多视角物体 3DGS、文本到 3D SDS、单图到 3D Zero123。最重要的修正是把旧点云/占位图全部替换为真实渲染输出，并把失败实验如实记录为调参对比。3DGS 部分已经达到较稳定的官方风格结果；B/C 由于是单机短训，质量低于官网长训 gallery，但已经具备可辨认物体、测试视角序列、三联诊断图和完整日志，满足作业复现和结果分析的基本要求。", "body"))
    elems.append(P("后续若要进一步提升质量，建议：1）B 使用 ProlificDreamer/VSD 或 MVDream 多视角扩散先验；2）C 下载原版 Zero123-XL 或 Stable-Zero123 权重并延长到 3000+ steps；3）融合阶段改为统一 mesh/gaussian 空间中的尺度配准和光照一致化。", "body"))

    return elems


def main() -> None:
    pdf_path = OUT / "HW3_Task1_3DGS_AIGC_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
    )
    doc.build(story(), onFirstPage=page_footer, onLaterPages=page_footer)
    print(pdf_path)


if __name__ == "__main__":
    main()
