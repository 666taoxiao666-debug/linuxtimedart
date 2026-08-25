#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成课题组组会汇报 PPT（5页，16:9，中英文混排）"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ── 路径 ──
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PPT_PATH = OUT_DIR / "PromptTimeDART_SDWPF_组会汇报.pptx"

BEST_RUN_DIR = (
    ROOT
    / "outputs/test_results/PromptTimeDART/SDWPF/"
    "MS_il336_pl96_dm128_el2_p12_s12_lossMIXED_lr3e-05_res1_hw1.0_pw0.0_mix0.8_seed2024_id20260804_113342"
)
BLEND_DIR = ROOT / "outputs/blend_results/PromptTimeDART/SDWPF/id20260804_124707"
ASSETS = ROOT / "assets"

# ── 配色（参考 Verifiable AI 风格）──
NAVY = RGBColor(0, 51, 102)
ROYAL = RGBColor(30, 90, 168)
SKY = RGBColor(232, 240, 248)
LIGHT_BLUE = RGBColor(210, 228, 245)
WHITE = RGBColor(255, 255, 255)
GOLD = RGBColor(245, 166, 35)
GRAY = RGBColor(100, 110, 125)
DARK = RGBColor(40, 50, 65)
ACCENT_GREEN = RGBColor(46, 139, 87)

FONT_CN = "Noto Sans CJK SC"
FONT_EN = "Arial"

# ── 10 次训练数据 ──
RUNS = [
    {"id": 1, "mae": None, "rmse": None, "r2": None, "label": "MSE\n无残差"},
    {"id": 2, "mae": 301.22, "rmse": 396.96, "r2": -0.112, "label": "Huber\nlr=3e-5"},
    {"id": 3, "mae": 301.22, "rmse": 396.96, "r2": -0.112, "label": "Huber\n(复现)"},
    {"id": 4, "mae": 276.32, "rmse": 373.92, "r2": 0.014, "label": "MSE\n+残差"},
    {"id": 5, "mae": 273.47, "rmse": 372.14, "r2": 0.023, "label": "MIXED\n+残差"},
    {"id": 6, "mae": 278.79, "rmse": 376.22, "r2": 0.001, "label": "MIXED\npw=0.5"},
    {"id": 7, "mae": 274.77, "rmse": 373.70, "r2": 0.015, "label": "MIXED\nhw=1.5"},
    {"id": 8, "mae": 273.47, "rmse": 372.14, "r2": 0.023, "label": "MIXED\n复现"},
    {"id": 9, "mae": 272.24, "rmse": 365.55, "r2": 0.057, "label": "MIXED\nlr=3e-5★"},
    {"id": 10, "mae": 273.42, "rmse": 366.69, "r2": 0.051, "label": "MIXED\nhw=1.5"},
]
PERSIST = {"mae": 222.54, "rmse": 369.49, "r2": 0.037}
BEST_MODEL = {"mae": 272.24, "rmse": 365.55, "r2": 0.057}
BEST_BLEND = {"mae": 232.47, "rmse": 343.40, "r2": 0.168}


def _set_font(run, size=14, bold=False, color=DARK, cn=True):
    run.font.name = FONT_EN
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if cn:
        run.font._element.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
            FONT_CN,
        )


def add_bg(slide, color=WHITE):
    bg = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    sp = slide.shapes._spTree
    sp.remove(bg._element)
    sp.insert(2, bg._element)


def add_header_bar(slide, title_cn, title_en, section=""):
    bar = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.08)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ROYAL
    bar.line.fill.background()

    if section:
        sec = slide.shapes.add_textbox(Inches(11.5), Inches(0.15), Inches(1.6), Inches(0.8))
        tf = sec.text_frame
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = section
        _set_font(r, 36, True, RGBColor(220, 228, 238), cn=False)

    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12), Inches(0.7))
    tf = box.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title_cn
    _set_font(r, 22, True, NAVY)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = title_en
    _set_font(r2, 11, False, GRAY, cn=False)


def add_card(slide, left, top, width, height, title, bullets, icon_color=ROYAL):
    card = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    card.fill.solid()
    card.fill.fore_color.rgb = SKY
    card.line.color.rgb = LIGHT_BLUE
    card.line.width = Pt(1)

    icon = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.OVAL, left + Inches(0.15), top + Inches(0.15),
        Inches(0.28), Inches(0.28),
    )
    icon.fill.solid()
    icon.fill.fore_color.rgb = icon_color
    icon.line.fill.background()

    tb = slide.shapes.add_textbox(
        left + Inches(0.15), top + Inches(0.5), width - Inches(0.3), height - Inches(0.55)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    _set_font(r, 12, True, NAVY)
    for b in bullets:
        p = tf.add_paragraph()
        p.space_before = Pt(4)
        r = p.add_run()
        r.text = "▸ " + b
        _set_font(r, 10, False, DARK)


def add_summary_box(slide, text, top=Inches(6.3)):
    box = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(0.5), top, Inches(12.333), Inches(0.85),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(230, 242, 255)
    box.line.color.rgb = ROYAL
    box.line.width = Pt(1.5)
    tb = slide.shapes.add_textbox(Inches(0.7), top + Inches(0.12), Inches(11.9), Inches(0.65))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "💡 核心结论 | Key Takeaway："
    _set_font(r, 11, True, NAVY)
    r2 = p.add_run()
    r2.text = text
    _set_font(r2, 11, False, DARK)


def add_nav_bar(slide, active_idx):
    sections = ["背景", "问题与设计", "方法", "结果", "总结"]
    w = Inches(12.333) / len(sections)
    for i, name in enumerate(sections):
        left = Inches(0.5) + w * i
        rect = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE, left, Inches(7.05), w, Inches(0.35)
        )
        if i == active_idx:
            rect.fill.solid()
            rect.fill.fore_color.rgb = ROYAL
            tc = WHITE
        else:
            rect.fill.solid()
            rect.fill.fore_color.rgb = SKY
            tc = GRAY
        rect.line.fill.background()
        tb = slide.shapes.add_textbox(left, Inches(7.08), w, Inches(0.3))
        tf = tb.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = f"0{i+1} {name}"
        _set_font(r, 8, i == active_idx, tc)


def add_table(slide, left, top, width, rows_data, col_widths=None, header=True):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    height = Inches(0.35 * n_rows)
    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for ri, row in enumerate(rows_data):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    if header and ri == 0:
                        _set_font(r, 9, True, WHITE)
                    elif ci == 0:
                        _set_font(r, 9, True, NAVY)
                    else:
                        _set_font(r, 9, False, DARK)
            if header and ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = SKY
    return table


def add_image_safe(slide, path, left, top, width=None, height=None):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)
        return True
    return False


def make_training_chart():
    """生成 10 次训练 MAE/RMSE 对比图"""
    fig, ax1 = plt.subplots(figsize=(10, 4.2), dpi=150)
    ids = [r["id"] for r in RUNS if r["mae"] is not None]
    maes = [r["mae"] for r in RUNS if r["mae"] is not None]
    rmses = [r["rmse"] for r in RUNS if r["mae"] is not None]

    x = np.arange(len(ids))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, maes, w, label="MAE (kW)", color="#1E5AA8", alpha=0.85)
    bars2 = ax1.bar(x + w / 2, rmses, w, label="RMSE (kW)", color="#4A90D9", alpha=0.75)
    ax1.axhline(PERSIST["mae"], color="#F5A623", ls="--", lw=1.5, label=f"Persistence MAE={PERSIST['mae']}")
    ax1.axhline(PERSIST["rmse"], color="#E74C3C", ls=":", lw=1.5, label=f"Persistence RMSE={PERSIST['rmse']}")

    best_idx = ids.index(9)
    bars1[best_idx - 2].set_edgecolor("#F5A623")
    bars1[best_idx - 2].set_linewidth(2.5)
    bars2[best_idx - 2].set_edgecolor("#F5A623")
    bars2[best_idx - 2].set_linewidth(2.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Run {i}" for i in ids], fontsize=8)
    ax1.set_ylabel("Error (kW)", fontsize=10)
    ax1.set_title("10-Run Training Progression | 十次训练误差演进", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(200, 420)
    plt.tight_layout()
    path = OUT_DIR / "chart_10runs.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_horizon_chart():
    """生成按预测时段的 Skill 对比图"""
    segments = ["Short\n1-12", "Mid-Short\n13-24", "Mid\n25-48", "Long\n49-96"]
    model_skill = [-5.8, -10.8, -9.9, 5.7]
    blend_skill = [0.0, 0.0, 1.9, 9.4]

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w / 2, model_skill, w, label="Model (Run 9)", color="#1E5AA8")
    ax.bar(x + w / 2, blend_skill, w, label="Blend (Best)", color="#2E8B57")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, fontsize=9)
    ax.set_ylabel("RMSE Skill vs Persistence (%)", fontsize=9)
    ax.set_title("Horizon-wise RMSE Skill | 分时段 RMSE Skill", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / "chart_horizon_skill.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_method_flowchart():
    """生成研究流程示意图"""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, "SDWPF\nData\n144K train", "#E8F0F8"),
        (2.0, 1.0, "Self-Supervised\nPretrain\nDiffusion Loss", "#1E5AA8"),
        (3.8, 1.0, "PromptTimeDART\nFinetune\nMIXED+Residual", "#1E5AA8"),
        (5.6, 1.0, "Test Eval\n50K windows\nMAE/RMSE/R²", "#4A90D9"),
        (7.4, 1.0, "Persistence\nBlend\nCutoff=24", "#2E8B57"),
        (9.0, 1.0, "Best\nRMSE Skill\n+7.1%", "#F5A623"),
    ]
    for i, (x, y, txt, c) in enumerate(boxes):
        fc = c if i in (0, 5) else c
        text_color = "white" if i in (1, 2, 3, 4) else "#003366"
        rect = plt.Rectangle((x, y), 1.4, 1.0, fc=fc, ec="#003366", lw=1.2, zorder=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.7, y + 0.5, txt, ha="center", va="center", fontsize=7,
                color=text_color, fontweight="bold")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0], 1.5), xytext=(x + 1.45, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#003366", lw=1.5))

    ax.set_title("Research Pipeline | 研究设计流程", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    path = OUT_DIR / "chart_pipeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def slide1_title(prs):
    """第1页：标题与研究背景"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)

    # 左侧装饰条
    stripe = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(0.12), Inches(7.5)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = ROYAL
    stripe.line.fill.background()

    # 标题
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.8), Inches(12), Inches(1.8))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "PromptTimeDART：基于扩散自回归 Transformer 的风电功率预测"
    _set_font(r, 28, True, NAVY)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "Wind Power Forecasting via Diffusion Autoregressive Transformer on SDWPF"
    _set_font(r2, 14, False, GRAY, cn=False)

    p3 = tf.add_paragraph()
    r3 = p3.add_run()
    r3.text = "TimeDART (ICML 2025) 扩展 · 10 次系统训练 · SDWPF 基准"
    _set_font(r3, 11, False, ROYAL)

    # 三栏背景卡片
    add_card(slide, Inches(0.5), Inches(2.5), Inches(3.9), Inches(2.2),
             "研究背景 | Background", [
                 "全球风电装机快速增长，功率预测是电网调度核心",
                 "TimeDART (Wang et al., ICML 2025) 统一因果 Transformer 与扩散去噪",
                 "SDWPF：多风机、10min 分辨率、强非平稳性",
             ])
    add_card(slide, Inches(4.7), Inches(2.5), Inches(3.9), Inches(2.2),
             "核心挑战 | Challenges", [
                 "Persistence 基线短期极强 (MAE≈223 kW)",
                 "功率 ramp 事件与低/高功率段预测偏差大",
                 "纯深度学习模型长期段才有优势",
             ], ACCENT_GREEN)
    add_card(slide, Inches(8.9), Inches(2.5), Inches(3.9), Inches(2.2),
             "本工作目标 | Objectives", [
                 "扩展 PromptTimeDART 至 SDWPF 下游预测",
                 "系统 10 次训练寻优损失/学习率/权重策略",
                 "Persistence 融合实现全局最优 RMSE Skill +7.1%",
             ], GOLD)

    # TimeDART 架构缩略图
    add_image_safe(slide, ASSETS / "2_TimeDART.png", Inches(0.5), Inches(4.95), height=Inches(1.8))

    add_summary_box(slide,
        "在 ICML 2025 TimeDART 框架上引入工况 Prompt 与残差预测，"
        "经 10 轮迭代训练 + Persistence 融合，RMSE 较基线提升 7.1%。")

    add_nav_bar(slide, 0)


def slide2_problem_design(prs, pipeline_chart):
    """第2页：科学问题与研究设计"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header_bar(slide, "科学问题与研究设计", "Scientific Questions & Research Design", "01")

    # 科学问题
    add_card(slide, Inches(0.5), Inches(1.1), Inches(6.0), Inches(1.5),
             "科学问题 | Research Questions", [
                 "Q1: TimeDART 自监督表征能否迁移至 SDWPF 多变量风电预测？",
                 "Q2: 工况 Prompt + 残差头 + MIXED 损失能否改善 ramp/长期段？",
                 "Q3: 如何融合 Persistence 短期优势与模型长期 Skill？",
             ])

    # 数据集统计
    add_table(slide, Inches(6.8), Inches(1.1), Inches(6.0),
              [
                  ["数据集参数", "数值", "说明"],
                  ["输入/预测长度", "336 / 96", "56h → 16h @10min"],
                  ["特征通道", "10 → 1 (MS)", "风速/风向/温度/桨距角→功率"],
                  ["训练/验证/测试", "144K / 21K / 51K", "70/10/20 时间切分"],
                  ["Persistence 基线", "MAE 222.5 kW", "RMSE 369.5 kW, R²=0.037"],
              ],
              [Inches(2.0), Inches(1.8), Inches(2.2)])

    # 流程图
    add_image_safe(slide, pipeline_chart, Inches(0.5), Inches(2.85), width=Inches(12.3))

    # 10次实验矩阵
    add_table(slide, Inches(0.5), Inches(4.5), Inches(12.333),
              [
                  ["Run", "损失", "关键变量", "MAE↓", "RMSE↓", "R²"],
                  ["1-3", "MSE/Huber", "无残差/Huber", "301", "397", "−0.11"],
                  ["4", "MSE", "+残差预测", "276", "374", "0.01"],
                  ["5-8", "MIXED", "lr=1e-4, 复现", "273", "372", "0.02"],
                  ["9★", "MIXED", "lr=3e-5", "272", "366", "0.06"],
                  ["10", "MIXED", "hw=1.5", "273", "367", "0.05"],
                  ["Blend★", "后处理", "cutoff=24, β=1", "232", "343", "0.17"],
              ],
              [Inches(0.7), Inches(1.5), Inches(2.5), Inches(1.2), Inches(1.2), Inches(1.0)])

    add_summary_box(slide,
        "残差预测 (Run 4) 与 MIXED 损失 (Run 5) 使 MAE 从 301→273 kW；"
        "学习率降至 3e-5 (Run 9) 首次 RMSE 超越 Persistence。")

    add_nav_bar(slide, 1)


def slide3_methods(prs):
    """第3页：方法与机制模型"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header_bar(slide, "方法：PromptTimeDART 机制", "Method: PromptTimeDART Architecture", "02")

    # 左侧四模块卡片
    modules = [
        ("① 实例归一化 + Patch", "336步→28 patches\npatch_len=12, stride=12"),
        ("② 因果 Transformer 编码", "d_model=128, 2层\n全局时序依赖建模"),
        ("③ 工况 Prompt 模块", "Regime Predictor\nSoft Prompt Generator\nAdaLN 去噪解码"),
        ("④ 残差预测头", "预测 Δpower\n零初始化 head\nMIXED Loss (80%MSE+20%MAE)"),
    ]
    for i, (title, desc) in enumerate(modules):
        y = Inches(1.1) + Inches(1.15) * i
        card = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(0.5), y, Inches(4.5), Inches(1.0),
        )
        card.fill.solid()
        card.fill.fore_color.rgb = SKY if i % 2 == 0 else WHITE
        card.line.color.rgb = LIGHT_BLUE
        tb = slide.shapes.add_textbox(Inches(0.65), y + Inches(0.1), Inches(4.2), Inches(0.85))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        _set_font(r, 11, True, NAVY)
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = desc
        _set_font(r2, 9, False, DARK)

    # 右侧架构图
    add_image_safe(slide, ASSETS / "2_TimeDART.png", Inches(5.3), Inches(1.1), width=Inches(7.5))

    # 融合机制
    add_card(slide, Inches(0.5), Inches(5.7), Inches(12.333), Inches(0.55),
             "Persistence 融合策略 | Blending Mechanism", [
                 "blend[h] = (1−w[h])·Persistence + w[h]·Model  |  前24步 w=0 (100% Persist)，25~96步线性过渡至 w=1",
             ], ACCENT_GREEN)

    add_summary_box(slide,
        "PromptTimeDART 在 TimeDART 扩散预训练表征上叠加工况感知 Prompt 与残差预测；"
        "融合策略利用「短期 Persist + 长期 Model」互补优势。", top=Inches(6.35))

    add_nav_bar(slide, 2)


def slide4_results(prs, chart_10runs, chart_horizon):
    """第4页：关键结果与数据可视化"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header_bar(slide, "关键结果与数据可视化", "Key Results & Data Visualization", "03")

    # 三方对比表
    add_table(slide, Inches(0.5), Inches(1.05), Inches(5.5),
              [
                  ["方法", "MAE (kW)", "RMSE (kW)", "R²", "RMSE Skill"],
                  ["Persistence", "222.54", "369.49", "0.037", "—"],
                  ["Model (Run 9★)", "272.24", "365.55", "0.057", "+1.1%"],
                  ["Blend (全局最佳★)", "232.47", "343.40", "0.168", "+7.1%"],
              ],
              [Inches(1.5), Inches(1.0), Inches(1.0), Inches(0.8), Inches(1.2)])

    # 10次训练图
    add_image_safe(slide, chart_10runs, Inches(6.3), Inches(1.0), width=Inches(6.5))

    # 分时段 Skill 图
    add_image_safe(slide, chart_horizon, Inches(0.5), Inches(3.0), width=Inches(5.5))

    # 嵌入真实实验图
    add_image_safe(slide, BEST_RUN_DIR / "error_by_horizon.png",
                   Inches(6.3), Inches(3.0), width=Inches(3.2))
    add_image_safe(slide, BEST_RUN_DIR / "prediction_scatter.png",
                   Inches(9.6), Inches(3.0), width=Inches(3.2))
    add_image_safe(slide, BLEND_DIR / "forecast_examples.png",
                   Inches(0.5), Inches(4.85), width=Inches(6.0))
    add_image_safe(slide, BLEND_DIR / "blend_weights.png",
                   Inches(6.8), Inches(4.85), width=Inches(6.0))

    add_summary_box(slide,
        "Run 9 纯模型 RMSE 365.55 kW 首次优于 Persistence；"
        "Blend 全局最佳 RMSE 343.40 kW (Skill +7.1%)，长期段 Skill +9.4%。",
        top=Inches(6.55))

    add_nav_bar(slide, 3)


def slide5_summary(prs):
    """第5页：创新点、讨论与展望"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_header_bar(slide, "创新点、讨论与展望", "Innovation, Discussion & Future Work", "04")

    # 创新点表
    add_table(slide, Inches(0.5), Inches(1.05), Inches(6.2),
              [
                  ["维度", "本工作创新", "对比基线/文献"],
                  ["模型", "PromptTimeDART 工况 Prompt", "TimeDART (ICML'25)"],
                  ["预测", "零初始化残差头", "直接绝对值预测"],
                  ["损失", "MIXED + 可选 horizon/power 权重", "单一 MSE/Huber"],
                  ["数据", "SDWPF 严格时间切分", "避免跨风机泄露"],
                  ["后处理", "自适应 Persist-Model 融合", "纯模型或纯 Persist"],
              ],
              [Inches(1.0), Inches(2.5), Inches(2.7)])

    # 讨论与局限
    add_card(slide, Inches(7.0), Inches(1.05), Inches(5.8), Inches(2.0),
             "讨论与局限 | Discussion", [
                 "MAE 仍劣于 Persistence（+4.5%），短期段模型不如基线",
                 "低功率高估 (MBE=+97.5 kW)、高功率低估，ramp-down 失败",
                 "风机异质性强（TurbID 3 最差），需个性化建模",
                 "早停于 Epoch 1，存在过拟合风险",
             ], RGBColor(180, 80, 80))

    # 展望
    add_card(slide, Inches(7.0), Inches(3.25), Inches(5.8), Inches(1.8),
             "未来展望 | Future Work", [
                 "引入 NWP 数值天气预报与空间邻域信息",
                 "分风机/分工况微调 + 功率分段加权损失",
                 "探索 Quantile/扩散式概率预测，量化不确定性",
                 "在线自适应融合权重，部署于实际风电场",
             ], ACCENT_GREEN)

    # 核心数据亮点
    highlights = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(3.25), Inches(6.2), Inches(1.8),
    )
    highlights.fill.solid()
    highlights.fill.fore_color.rgb = RGBColor(255, 248, 230)
    highlights.line.color.rgb = GOLD
    highlights.line.width = Pt(2)

    metrics = [
        ("272.24", "kW MAE\n(纯模型最佳)"),
        ("343.40", "kW RMSE\n(融合最佳)"),
        ("+7.1%", "RMSE Skill\nvs Persist"),
        ("+9.4%", "长期段 Skill\n(49-96步)"),
    ]
    for i, (num, label) in enumerate(metrics):
        x = Inches(0.7) + Inches(1.5) * i
        circle = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.OVAL, x, Inches(3.45), Inches(1.2), Inches(1.2),
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = ROYAL if i < 2 else ACCENT_GREEN
        circle.line.fill.background()
        tb = slide.shapes.add_textbox(x, Inches(3.55), Inches(1.2), Inches(1.0))
        tf = tb.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = num
        _set_font(r, 16, True, WHITE, cn=False)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = label
        _set_font(r2, 7, False, WHITE)

    add_summary_box(slide,
        "本工作系统验证了 TimeDART→SDWPF 迁移路径，"
        "PromptTimeDART + 残差 + MIXED + 融合构成当前最优方案；"
        "下一步聚焦概率预测与风机个性化。",
        top=Inches(5.35))

    # 参考文献
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(6.35), Inches(12), Inches(0.6))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ("References: Wang et al., TimeDART, ICML 2025 (arXiv:2410.05711) · "
              "PatchTST (NeurIPS 2023) · SDWPF Wind Power Dataset")
    _set_font(r, 8, False, GRAY, cn=False)

    add_nav_bar(slide, 4)


def main():
    print("生成图表...")
    chart_10runs = make_training_chart()
    chart_horizon = make_horizon_chart()
    pipeline_chart = make_method_flowchart()

    print("生成 PPT...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide1_title(prs)
    slide2_problem_design(prs, pipeline_chart)
    slide3_methods(prs)
    slide4_results(prs, chart_10runs, chart_horizon)
    slide5_summary(prs)

    prs.save(str(PPT_PATH))
    print(f"✅ PPT 已保存: {PPT_PATH}")
    print(f"   辅助图表: {OUT_DIR}")


if __name__ == "__main__":
    main()
