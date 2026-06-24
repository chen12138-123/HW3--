from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
THESIS_FIG = FIG / "thesis"
FORMULA_FIG = THESIS_FIG / "formulas"
REPORT_DIR = ROOT / "report"

PROJECT_GITHUB = "https://github.com/chen12138-123/HW3--"
GROUP_MEMBERS = "周湘洋、毛琦骏、陈希"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def register_fonts() -> tuple[str, str, str, str]:
    simsun = Path(r"C:\Windows\Fonts\simsun.ttc")
    simhei = Path(r"C:\Windows\Fonts\simhei.ttf")
    times = Path(r"C:\Windows\Fonts\times.ttf")
    times_bold = Path(r"C:\Windows\Fonts\timesbd.ttf")
    cn = "Helvetica"
    cn_bold = "Helvetica-Bold"
    en = "Times-Roman"
    en_bold = "Times-Bold"
    if simsun.exists():
        pdfmetrics.registerFont(TTFont("SimSun", str(simsun)))
        cn = "SimSun"
    if simhei.exists():
        pdfmetrics.registerFont(TTFont("SimHei", str(simhei)))
        cn_bold = "SimHei"
    if times.exists():
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(times)))
        en = "TimesNewRoman"
    if times_bold.exists():
        pdfmetrics.registerFont(TTFont("TimesNewRomanBold", str(times_bold)))
        en_bold = "TimesNewRomanBold"
    return cn, cn_bold, en, en_bold


CN_FONT, CN_BOLD, EN_FONT, EN_BOLD = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=base["BodyText"],
        fontName=CN_FONT,
        fontSize=12,
        leading=18,
        firstLineIndent=24,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceAfter=6,
    )
    return {
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
        "author": ParagraphStyle(
            "author",
            parent=base["BodyText"],
            fontName=CN_FONT,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceAfter=4,
        ),
        "abstract_title": ParagraphStyle(
            "abstract_title",
            parent=base["Heading2"],
            fontName=CN_BOLD,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            wordWrap="CJK",
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=CN_BOLD,
            fontSize=14,
            leading=20,
            leftIndent=0,
            firstLineIndent=0,
            wordWrap="CJK",
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=CN_BOLD,
            fontSize=12,
            leading=18,
            leftIndent=0,
            firstLineIndent=0,
            wordWrap="CJK",
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": body,
        "body_no_indent": ParagraphStyle(
            "body_no_indent",
            parent=body,
            firstLineIndent=0,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=CN_FONT,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            wordWrap="CJK",
            textColor=colors.HexColor("#303030"),
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=CN_FONT,
            fontSize=9,
            leading=13,
            firstLineIndent=0,
            alignment=TA_LEFT,
            wordWrap="CJK",
            textColor=colors.HexColor("#303030"),
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName=EN_FONT,
            fontSize=8.8,
            leading=12,
            leftIndent=10,
            rightIndent=10,
            firstLineIndent=0,
            backColor=colors.HexColor("#f5f5f5"),
            borderColor=colors.HexColor("#d8d8d8"),
            borderWidth=0.45,
            borderPadding=5,
            wordWrap="CJK",
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


S = make_styles()


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def en(text: object, bold: bool = False) -> str:
    return f'<font name="{EN_BOLD if bold else EN_FONT}">{esc(text)}</font>'


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def plain(text: object, style: str = "body") -> Paragraph:
    return p(esc(text), style)


def h1(index: str, text: str) -> Paragraph:
    return p(f"{en(index, True)}&nbsp;&nbsp;{esc(text)}", "h1")


def h2(index: str, text: str) -> Paragraph:
    return p(f"{en(index, True)}&nbsp;&nbsp;{esc(text)}", "h2")


def noindent(text: str) -> Paragraph:
    return p(text, "body_no_indent")


def code(text: str) -> Paragraph:
    return plain(text, "code")


def file_size(path: Path) -> str:
    if not path.exists():
        return "未在本地找到"
    mb = path.stat().st_size / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def image_flow(path: Path, caption: str, width: float = 15.8 * cm, max_height: float = 16.8 * cm) -> list[Any]:
    if not path.exists():
        return [plain(f"缺少图片：{path}", "small")]
    with PILImage.open(path) as im:
        iw, ih = im.size
    draw_w = width
    draw_h = draw_w * ih / iw
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h * iw / ih
    return [
        Spacer(1, 3),
        Image(str(path), width=draw_w, height=draw_h, hAlign="CENTER"),
        plain(caption, "caption"),
    ]


def fig(name: str, caption: str, width: float = 15.8 * cm, max_height: float = 16.8 * cm) -> list[Any]:
    return image_flow(THESIS_FIG / name, caption, width, max_height)


def oldfig(name: str, caption: str, width: float = 15.8 * cm, max_height: float = 16.8 * cm) -> list[Any]:
    return image_flow(FIG / name, caption, width, max_height)


def formula(name: str, caption: str, width: float = 13.6 * cm) -> list[Any]:
    return image_flow(FORMULA_FIG / f"{name}.png", caption, width, 3.0 * cm)


def run_figure_builder() -> None:
    script = ROOT / "src" / "build_thesis_figures.py"
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)


def build_formula_figures() -> None:
    FORMULA_FIG.mkdir(parents=True, exist_ok=True)
    formulas = {
        "gaussian": r"$G_i=\{\mu_i,\Sigma_i,\alpha_i,\mathbf{c}_i\},\qquad \Sigma_i=R_iS_iS_i^{\mathsf{T}}R_i^{\mathsf{T}}$",
        "render": r"$C(\mathbf{r})=\sum_{i\in\mathcal{N}(\mathbf{r})}T_i\alpha_i\mathbf{c}_i,\qquad T_i=\prod_{j<i}(1-\alpha_j)$",
        "loss": r"$\mathcal{L}_{3DGS}=(1-\lambda)|I-\hat I|_1+\lambda(1-\mathrm{SSIM}(I,\hat I)),\quad \lambda=0.2$",
        "sds": r"$\nabla_\theta\mathcal{L}_{SDS}=\mathbb{E}_{t,\epsilon}\!\left[w(t)\left(\epsilon_\phi(x_t;y,t)-\epsilon\right)\frac{\partial x}{\partial\theta}\right]$",
        "zero123": r"$\mathcal{L}_{img2D}=\mathbb{E}_{t,\epsilon,\Delta\pi}\left|\epsilon_\phi(x_t;I_0,\Delta\pi,t)-\epsilon\right|_2^2$",
        "sh0": r"$\mathbf{f}_{dc}=\frac{\mathbf{rgb}/255-0.5}{C_0},\qquad C_0=0.2820947918$",
        "fusion": r"$\mathbf{p}'=sR_z(\theta)\mathbf{p}+\mathbf{t},\qquad \mathcal{V}_{fused}=\mathcal{V}_{bg}\cup T_A(\mathcal{V}_A)\cup T_B(\mathcal{V}_B)\cup T_C(\mathcal{V}_C)$",
        "metrics": r"$\operatorname{PSNR}=-10\log_{10}(\operatorname{MSE}),\qquad \operatorname{L1}=\frac{1}{3N}\sum_{u,v,c}|I_{uvc}-\hat I_{uvc}|$",
    }
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    for name, expr in formulas.items():
        path = FORMULA_FIG / f"{name}.png"
        fig_obj = plt.figure(figsize=(9.5, 1.0), dpi=220)
        ax = fig_obj.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.52, expr, ha="center", va="center", fontsize=17, color="#111111")
        fig_obj.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig_obj)


def draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setFont(EN_FONT, 8.5)
    canvas.setFillColor(colors.HexColor("#333333"))
    canvas.drawRightString(width - 1.7 * cm, 0.72 * cm, str(doc.page))
    canvas.restoreState()


def front_matter() -> list[Any]:
    elems: list[Any] = []
    elems.append(Spacer(1, 0.55 * cm))
    elems.append(HRFlowable(width="100%", thickness=4.0, color=colors.black, spaceBefore=0, spaceAfter=22))
    elems.append(plain("期末作业题目一：\n基于 3DGS 与 AIGC 的多源资产生成与真实场景融合", "title"))
    elems.append(HRFlowable(width="100%", thickness=1.0, color=colors.black, spaceBefore=2, spaceAfter=22))
    elems.append(plain(GROUP_MEMBERS, "author"))
    elems.append(p(f"课程：深度学习与空间智能&nbsp;&nbsp;&nbsp;&nbsp;提交仓库：{en(PROJECT_GITHUB)}", "author"))
    elems.append(p("正文中文采用宋体小四，英文与公式采用 Times New Roman 风格；图表数据均以可视化图片呈现。", "author"))
    elems.append(Spacer(1, 12))
    elems.append(plain("摘 要", "abstract_title"))
    elems.append(
        p(
            f"本报告完成题目一的全链路三维空间智能实验：以开源 {en('Mip-NeRF 360 garden')} 场景作为统一背景，使用 {en('GraphDECO 3D Gaussian Splatting')} 完成真实场景重建；物体 A 使用真实多视角 {en('Tanks and Temples truck')} 数据复现 {en('COLMAP + 3DGS')} 路线；物体 B 依据题目要求使用 {en('threestudio DreamFusion/SDS')} 从文本提示生成三维资产，并补充本机可完整交付的先进前馈三维生成结果；物体 C 使用单张去背景前景图、{en('Zero123')} 视角先验和 {en('TripoSR')} 生成可融合三维资产。最终阶段不再采用截图贴片，而是在三维数据层读取背景、A、B、C 四路 PLY 或 3DGS-compatible PLY，执行尺度、旋转、平移、颜色过滤与顶点字段级拼接，输出融合后的 {en('model.ply')} 和多视角漫游视频。实验显示，真实多视角 3DGS 在几何一致性和可评价性上最稳定，背景测试集 PSNR 为 {en('30.4103 dB')}，物体 A 测试集 PSNR 为 {en('26.1683 dB')}；文本到 3D 和单图到 3D 具有更低输入成本，但更依赖扩散先验、提示词和视角补全能力。"
        )
    )
    return elems


def story() -> list[Any]:
    asset_meta = load_json(ROOT / "output" / "final_assets" / "final_asset_summary.json")
    fusion_meta = load_json(ROOT / "output" / "final_fused" / "fusion_metadata.json")
    b_meta = asset_meta.get("object_b", {})
    c_meta = asset_meta.get("object_c", {})
    comps = fusion_meta.get("components", {})

    elems: list[Any] = []
    elems.extend(front_matter())

    elems.append(h1("1", "任务定义与总体路线"))
    elems.append(
        p(
            "题目一要求在同一三维场景中组织真实重建资产与 AIGC 资产，因此本实验把工作拆成四条可验证链路：真实背景重建、真实多视角物体重建、文本到 3D 生成、单图到 3D 生成。旧版结果的主要问题是融合阶段过于接近二维截图叠加，缺少三维数据层证据。本版报告把融合定义为统一 PLY/vertex 表示后的三维坐标变换与字段拼接，并用融合后 PLY 的预览图、俯视图、点数统计和视频关键帧共同证明结果来源。"
        )
    )
    elems.extend(fig("08_pipeline_diagram.png", "图 1：题目一完整技术路线。所有分支均保留输入、训练或生成结果、可视化图与最终可融合资产。", max_height=11.5 * cm))
    elems.extend(fig("13_data_sources.png", "图 2：背景、物体 A、文本生成输入与单图输入的数据来源。", max_height=11.5 * cm))
    elems.append(
        p(
            "具体分工按实验环节组织：周湘洋负责 3DGS 背景重建、物体 A 多视角重建与训练指标整理；毛琦骏负责 task2 ACT 泛化实验、task1 的 B/C 生成路线调研与可视化对比；陈希负责三维融合、报告排版、README 与提交仓库整理。"
        )
    )

    elems.append(h1("2", "数学模型与关键公式"))
    elems.append(
        p(
            "为了补足报告中的理论部分，本节把后续实验使用的表示、渲染、优化、生成和融合目标统一写出。3DGS 把场景表示为一组各向异性三维高斯，每个高斯包含中心、协方差、不透明度和颜色；协方差由旋转矩阵和尺度矩阵分解，便于在训练中保持正定性。"
        )
    )
    elems.extend(formula("gaussian", "公式 1：3D Gaussian Splatting 的单个高斯参数与协方差分解。"))
    elems.append(
        p(
            "可微光栅化时，同一条相机射线上的高斯按照深度排序进行 alpha 合成，前方高斯通过透射率抑制后方高斯贡献。该过程使稀疏点云初始化能够逐步优化为可渲染的连续视图。"
        )
    )
    elems.extend(formula("render", "公式 2：沿视线方向的颜色合成模型。"))
    elems.append(
        p(
            "背景和物体 A 的训练损失沿用官方 3DGS 设置，由像素级 L1 损失和 DSSIM 损失组合而成。本实验在 7000 次迭代后导出 train/test 渲染结果，并以 L1 与 PSNR 进行评价。"
        )
    )
    elems.extend(formula("loss", "公式 3：3DGS 图像重建损失。"))
    elems.append(
        p(
            "物体 B 的课程指定路线是文本到 3D。SDS 通过冻结的二维扩散模型提供梯度，把随机视角渲染图朝文本提示对应的图像分布推进。由于梯度来自二维先验，短训练下容易出现多面纹理、形体塌缩或提示词敏感现象。"
        )
    )
    elems.extend(formula("sds", "公式 4：文本到 3D 的 Score Distillation Sampling 梯度形式。"))
    elems.append(
        p(
            "物体 C 的课程指定路线是单图到 3D。Zero123 类模型把输入图像和相对相机位姿作为条件，预测新视角扩散噪声；它比纯文本生成多了真实前景约束，但不可见背面的几何和纹理仍然依赖先验补全。"
        )
    )
    elems.extend(formula("zero123", "公式 5：单图条件新视角扩散目标。"))
    elems.append(
        p(
            "为把 B/C 的 mesh 输出并入 3DGS 背景，本实验把表面采样点颜色转换为球谐零阶颜色字段，随后写入与官方 PLY 接近的顶点结构。融合阶段对 A/B/C 执行刚体变换和尺度变换，最后拼接顶点集合。"
        )
    )
    elems.extend(formula("sh0", "公式 6：RGB 到 3DGS 球谐零阶颜色字段的转换。"))
    elems.extend(formula("fusion", "公式 7：三维坐标变换与数据级融合定义。"))
    elems.extend(formula("metrics", "公式 8：渲染质量评价指标。"))

    elems.append(h1("3", "环境、数据与复现实验设置"))
    elems.append(
        p(
            f"实验环境为 Windows + Anaconda，Python 3.10，PyTorch 2.1.2 + CUDA 11.8，GPU 为 NVIDIA GeForce RTX 4080 Laptop。官方 3DGS 使用 {en('submodules/gaussian-splatting')} 中的训练和渲染脚本；AIGC 路线使用 {en('submodules/threestudio')}、本地 Stable Diffusion v1.5 和 TripoSR 权重。Windows 下为了编译 nerfacc 与相关 CUDA 扩展，使用 Visual Studio 2019 环境，并在必要位置加入允许 unsupported compiler 的编译参数；这一修改只影响扩展编译，不改变算法目标。"
        )
    )
    elems.extend(fig("10_hyperparameters.png", "图 3：主要实验超参数、损失函数、训练步数、融合参数与输出规格。", max_height=12.3 * cm))
    elems.append(
        code(
            "python src/real_3dgs_pipeline.py --scene_name garden --iterations 7000 --resolution 4 --eval\n"
            "python src/real_3dgs_pipeline.py --scene_name object_a_truck --iterations 7000 --resolution 2 --eval\n"
            "python src/final_asset_pipeline.py\n"
            "python src/final_fusion.py\n"
            "python src/report_builder.py"
        )
    )
    elems.append(
        p(
            "仓库中不直接提交大型原始数据、预训练权重和大体积 PLY 结果，而是在 README 与提交说明中列出需要额外上传的文件。这样 GitHub 仓库保持为代码、报告、可视化结果和复现实验说明，权重与大文件通过云盘或课程平台单独提交。"
        )
    )

    elems.append(h1("4", "背景场景与物体 A 的 3DGS 重建"))
    elems.append(
        p(
            f"背景选择 Mip-NeRF 360 的 garden 场景，物体 A 使用官方真实多视角 truck 数据替代本地手机采集数据。二者均包含真实图像、COLMAP 相机参数和稀疏点云，因此能够完整复现“多视角输入、位姿估计、3DGS 训练、novel-view 渲染”的技术路线。训练 7000 次迭代后，背景 test L1 为 {en('0.0195519')}、test PSNR 为 {en('30.4103 dB')}；物体 A test L1 为 {en('0.0317515')}、test PSNR 为 {en('26.1683 dB')}。"
        )
    )
    elems.extend(fig("01_3dgs_loss_curves.png", "图 4：背景 garden 与物体 A truck 的 3DGS 训练损失曲线。", max_height=10.0 * cm))
    elems.extend(fig("02_3dgs_metrics.png", "图 5：背景与物体 A 的 train/test PSNR 和 L1 指标可视化。", max_height=10.0 * cm))
    elems.extend(oldfig("real_3dgs_garden_render_gt_compare.png", "图 6：背景 garden 的官方 3DGS render 与 GT 对比。", max_height=9.5 * cm))
    elems.extend(oldfig("real_3dgs_object_a_truck_render_gt_compare.png", "图 7：物体 A truck 的官方 3DGS render 与 GT 对比。", max_height=9.5 * cm))
    elems.append(
        p(
            "从结果看，garden 背景的视角覆盖更丰富，树木、地面和室外空间在测试视角上仍能保持较高保真度；truck 由于物体边缘、遮挡和反光区域更集中，PSNR 低于背景，但车体主体结构和颜色仍然稳定。"
        )
    )

    elems.append(PageBreak())
    elems.append(h1("5", "物体 B：文本到 3D 生成"))
    elems.append(
        p(
            "物体 B 首先按题目要求使用 threestudio DreamFusion/SDS 路线复现文本到 3D。早期复杂提示词“红色玩具跑车”在 2000 step 后仅形成红色团块，说明短训练 SDS 对细长结构和轮胎等复杂拓扑不稳定。最终课程路线保留苹果 prompt 的 1800 step 结果作为可解释的 SDS 成功样例，并用更稳健的前馈三维生成流程生成可融合资产。"
        )
    )
    elems.extend(oldfig("threestudio_b_sds_2000_rgb_views_grid.png", "图 8：复杂跑车 prompt 的失败对比，形体没有形成可靠车辆结构。", max_height=8.0 * cm))
    elems.extend(oldfig("threestudio_b_apple_sds_1800_progress_rgb.png", "图 9：物体 B 的 SDS 训练过程关键帧。", max_height=8.0 * cm))
    elems.extend(oldfig("threestudio_b_apple_sds_1800_rgb_views_grid.png", "图 10：物体 B 的 SDS 多视角测试渲染。", max_height=8.0 * cm))
    elems.extend(oldfig("object_b_final_input_rgba.png", "图 11：物体 B 最终进入 TripoSR 的去背景单物体输入。", width=7.0 * cm, max_height=6.5 * cm))
    elems.extend(oldfig("object_b_final_triposr_views_grid.png", "图 12：物体 B 最终 mesh 的多视角预览。", max_height=8.0 * cm))
    elems.append(
        p(
            f"最终 B 资产的 mesh 顶点数为 {en(b_meta.get('vertices', 0))}，面片数为 {en(b_meta.get('faces', 0))}，转换后的 3DGS-compatible splats 为 {en(b_meta.get('cloud_stats', {}).get('points', 0))}；本地生成耗时 {en(b_meta.get('elapsed_sec', 0))} 秒，峰值显存约 {en(b_meta.get('peak_memory_mb', 0))} MB。报告中同时保留 Hunyuan3D-2.1、TRELLIS/TRELLIS.2 和 Stable Fast 3D 的尝试状态，避免把未跑通的先进路线包装成完成结果。"
        )
    )

    elems.append(h1("6", "物体 C：单图到 3D 生成"))
    elems.append(
        p(
            "物体 C 使用去背景后的单张真实物体图像作为条件输入。Zero123 路线训练 600 step 后导出 60 个测试视角，能够保留汉堡正面颜色层次和主体体积，但不可见背面的纹理仍由生成先验补全。为了获得可提交、可融合的 mesh 与 PLY，本实验继续使用 TripoSR 输出最终资产。"
        )
    )
    elems.extend(oldfig("object_c_final_input_rgba.png", "图 13：物体 C 的去背景单图输入。", width=7.5 * cm, max_height=6.8 * cm))
    elems.extend(oldfig("threestudio_c_zero123_hamburger_600_progress_rgb.png", "图 14：物体 C 的 Zero123 训练过程关键帧。", max_height=8.0 * cm))
    elems.extend(oldfig("threestudio_c_zero123_hamburger_600_rgb_views_grid.png", "图 15：物体 C 的 Zero123 多视角测试渲染。", max_height=8.0 * cm))
    elems.extend(oldfig("object_c_final_triposr_views_grid.png", "图 16：物体 C 最终 mesh 的多视角预览。", max_height=8.0 * cm))
    elems.append(
        p(
            f"最终 C 资产的 mesh 顶点数为 {en(c_meta.get('vertices', 0))}，面片数为 {en(c_meta.get('faces', 0))}，转换后的 3DGS-compatible splats 为 {en(c_meta.get('cloud_stats', {}).get('points', 0))}；本地生成耗时 {en(c_meta.get('elapsed_sec', 0))} 秒，峰值显存约 {en(c_meta.get('peak_memory_mb', 0))} MB。与 B 相比，C 的输入约束更强，因此正面外观更接近输入图，但三维背面仍然存在先验补全带来的不确定性。"
        )
    )

    elems.append(PageBreak())
    elems.append(h1("7", "实验数据可视化与对比分析"))
    elems.append(
        p(
            "按照提交要求，本报告尽量避免把实验数据放成大表格，而是把关键指标、耗时、资产规模、先进算法尝试状态和质量比较都做成图片。需要特别说明的是，3DGS 的 L1/PSNR 来自有真实 GT 的渲染评价；B/C 没有多视角真实 GT，不能伪造 PSNR 或 SSIM，因此采用训练步数、资产几何规模、可视化质量、耗时和融合后点数进行比较。"
        )
    )
    elems.extend(fig("03_asset_stats.png", "图 17：B/C 最终资产的 mesh 顶点数、面片数、采样 splats 与生成耗时。", max_height=10.5 * cm))
    elems.extend(fig("04_compute_time.png", "图 18：主要训练和生成阶段的本机耗时对比。", max_height=9.5 * cm))
    elems.extend(fig("07_quality_radar.png", "图 19：三种生成路线的归一化质量对比。该图用于解释本实验现象，不代表公开 benchmark。", max_height=10.5 * cm))
    elems.extend(fig("11_sota_status.png", "图 20：Hunyuan3D-2.1、TRELLIS/TRELLIS.2、Stable Fast 3D 与 TripoSR 的尝试状态。", max_height=11.0 * cm))
    elems.append(
        p(
            "对比结果表明，真实多视角 3DGS 质量最稳定，缺点是数据采集或下载成本较高；文本到 3D 的输入成本最低，但短训练强依赖提示词和扩散先验；单图到 3D 在输入成本和可识别性之间取得折中，正面外观可靠性强于纯文本生成，但背面和厚度仍存在不确定性。"
        )
    )

    elems.append(h1("8", "三维数据级融合与漫游视频"))
    elems.append(
        p(
            f"最终融合读取四个输入：背景 garden 的官方 3DGS PLY、物体 A truck 的官方 3DGS PLY、物体 B 的 3DGS-compatible PLY、物体 C 的 3DGS-compatible PLY。融合元数据记录的方法为“{esc(fusion_meta.get('method', 'Raw 3DGS-compatible PLY merge after 3D coordinate transforms'))}”。融合后的总 splats/points 为 {en(comps.get('fused', {}).get('points', 0))}，其中背景采样 {en(comps.get('background', {}).get('points', 0))}，物体 A 采样 {en(comps.get('object_a', {}).get('points', 0))}，物体 B 过滤后 {en(comps.get('object_b', {}).get('points', 0))}，物体 C 过滤后 {en(comps.get('object_c', {}).get('points', 0))}。"
        )
    )
    elems.extend(fig("06_fusion_layout.png", "图 21：A/B/C 在背景坐标系中的插入位置俯视示意。", max_height=10.0 * cm))
    elems.extend(fig("05_fusion_point_counts.png", "图 22：最终融合场景的 splat/point 数量统计。", max_height=9.8 * cm))
    elems.extend(oldfig("final_fused_scene.png", "图 23：融合后 PLY 的三维预览图，不是背景截图加物体贴图。", max_height=9.6 * cm))
    elems.extend(oldfig("final_fused_scene_top.png", "图 24：融合后场景俯视图，用于检查空间尺度和相对位置。", max_height=9.0 * cm))
    elems.extend(fig("14_video_keyframes.png", "图 25：最终漫游视频 output/final_fused/roaming_video.mp4 的关键帧。", max_height=9.4 * cm))
    elems.append(
        p(
            "当前融合已经解决了二维截图贴片的问题，但仍存在光照一致性和接触阴影不足。原因是 B/C 由 mesh 表面采样并写入 SH0 颜色，缺少官方 3DGS 训练得到的高阶球谐和各向异性高斯参数。后续若继续提升质量，可把融合后的整体场景作为初始化做短程 3DGS finetune，或对 B/C 单独重训练高斯参数。"
        )
    )

    elems.append(PageBreak())
    elems.append(h1("9", "输出文件与提交说明"))
    elems.append(
        p(
            f"任务一报告、代码、README 和可视化图已整理到统一仓库 {en(PROJECT_GITHUB)}。大体积权重、原始数据和 PLY/GLB 结果不建议直接提交到 GitHub，应单独上传到云盘或课程平台，并在提交说明中写明下载链接。task1 当前本地应上传的核心结果包括：背景与物体 A 的 3DGS PLY、B/C 的 GLB 与 PLY、融合后的 PLY、漫游视频，以及必要时的 TripoSR 预训练权重。"
        )
    )
    elems.extend(fig("12_artifacts_manifest.png", "图 26：task1 模型权重与结果文件清单，供云盘或课程平台上传使用。", max_height=10.8 * cm))
    elems.append(
        code(
            "task1/report/main.pdf\n"
            "task1/output/real_3dgs_bg/point_cloud/iteration_7000/point_cloud.ply\n"
            "task1/output/object_a_official_3dgs/point_cloud/iteration_7000/point_cloud.ply\n"
            "task1/output/final_assets/object_b_final/mesh.glb\n"
            "task1/output/final_assets/object_b_final/model.ply\n"
            "task1/output/final_assets/object_c_final/mesh.glb\n"
            "task1/output/final_assets/object_c_final/model.ply\n"
            "task1/output/final_fused/model.ply\n"
            "task1/output/final_fused/roaming_video.mp4"
        )
    )
    elems.append(
        p(
            f"本地文件大小方面，B 的 GLB 为 {en(file_size(ROOT / 'output/final_assets/object_b_final/mesh.glb'))}，B 的 PLY 为 {en(file_size(ROOT / 'output/final_assets/object_b_final/model.ply'))}；C 的 GLB 为 {en(file_size(ROOT / 'output/final_assets/object_c_final/mesh.glb'))}，C 的 PLY 为 {en(file_size(ROOT / 'output/final_assets/object_c_final/model.ply'))}；融合 PLY 为 {en(file_size(ROOT / 'output/final_fused/model.ply'))}，漫游视频为 {en(file_size(ROOT / 'output/final_fused/roaming_video.mp4'))}。"
        )
    )

    elems.append(h1("10", "结论"))
    elems.append(
        p(
            "本次修订完成了题目一报告的完整闭环：重新对齐任务要求，补充了 3DGS、SDS、Zero123、SH0 转换、三维变换融合和评价指标公式；把实验数据、对比数据和输出清单尽量转为图片呈现；把作者信息、提交仓库和权重说明补全；最终融合明确采用三维数据级 PLY 拼接，不再把场景 A 截图作为融合结果。实验结论是：真实多视角 3DGS 仍是几何和视角一致性最强的路线，文本到 3D 适合低成本语义创作但短训练不稳定，单图到 3D 在输入成本和物体可识别性之间更均衡。"
        )
    )
    return elems


def build_report(root: Path = ROOT) -> Path:
    run_figure_builder()
    build_formula_figures()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / "main.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.35 * cm,
        title="HW3 Task 1 3DGS AIGC Report",
        author=GROUP_MEMBERS,
    )
    doc.build(story(), onFirstPage=draw_page, onLaterPages=draw_page)
    legacy = REPORT_DIR / "HW3_Report.pdf"
    shutil.copyfile(pdf_path, legacy)
    print(f"Report saved to {pdf_path}")
    print(f"Compatibility copy saved to {legacy}")
    return pdf_path


def main() -> None:
    build_report(ROOT)


if __name__ == "__main__":
    main()
