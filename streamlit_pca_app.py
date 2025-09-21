import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import tempfile
import base64
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# matplotlib 全体設定（論文体裁）
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 16,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# ReportLab 日本語フォント登録
try:
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
except Exception:
    pass

# 白色板スペクトルデータ（仮置き）
WHITE_BOARD_SPECTRUM = np.array([
    2824, 1152, 2967, 1046, 1477, 1899, 514, 543, 2768, 315, 545, 89, 174, 126, 243, 1390, 61, 32])

st.set_page_config(
    page_title="irodori 解析アプリ",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# カスタムCSSでアプリの見た目を改善
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# メインヘッダー
st.markdown("""
<div class="main-header">
    <h1>📊 irodori 解析アプリ</h1>
    <p>主成分分析（PCA）レポート作成アプリケーション</p>
</div>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'report_title' not in st.session_state:
    st.session_state.report_title = "PCA分析レポート"
if 'author_name' not in st.session_state:
    st.session_state.author_name = ""
if 'analysis_date' not in st.session_state:
    st.session_state.analysis_date = None
if 'mean_spectrum_notes' not in st.session_state:
    st.session_state.mean_spectrum_notes = ""
if 'pca_scatter_notes' not in st.session_state:
    st.session_state.pca_scatter_notes = ""
if 'loading_notes' not in st.session_state:
    st.session_state.loading_notes = ""
if 'overall_conclusion' not in st.session_state:
    st.session_state.overall_conclusion = ""
if 'corrected_spectrum_notes' not in st.session_state:
    st.session_state.corrected_spectrum_notes = ""
if 'label_colors' not in st.session_state:
    st.session_state.label_colors = {}
if 'current_labels' not in st.session_state:
    st.session_state.current_labels = []
if 'use_corrected_data_for_pca' not in st.session_state:
    st.session_state.use_corrected_data_for_pca = False
if 'selected_labels' not in st.session_state:
    st.session_state.selected_labels = []
if 'show_all_samples' not in st.session_state:
    st.session_state.show_all_samples = True

def load_and_prepare_data(df):
    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'class', 'category', 'group']:
            label_col = col
            break

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if label_col:
        labels = df[label_col].values
        feature_columns = [col for col in numeric_columns if col != label_col]
    else:
        labels = None
        feature_columns = numeric_columns

    features = df[feature_columns].values
    return features, labels, feature_columns, label_col

def get_label_colors(labels):
    """ラベルごとの色を取得（ユーザー設定またはデフォルト）"""
    if labels is None:
        return {}

    unique_labels = np.unique(labels)
    colors = {}

    # デフォルトカラーパレット
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for i, label in enumerate(unique_labels):
        if label in st.session_state.label_colors:
            colors[label] = st.session_state.label_colors[label]
        else:
            colors[label] = default_colors[i % len(default_colors)]

    return colors

def create_color_picker_ui(labels):
    """色選択UIを作成"""
    if labels is None:
        return

    unique_labels = np.unique(labels)

    # ラベルが変更された場合は色設定をリセット
    if st.session_state.current_labels != list(unique_labels):
        st.session_state.current_labels = list(unique_labels)
        st.session_state.label_colors = {}

    st.sidebar.markdown("### 🎨 グラフ色設定")

    # デフォルトカラーパレット
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for i, label in enumerate(unique_labels):
        default_color = default_colors[i % len(default_colors)]
        current_color = st.session_state.label_colors.get(label, default_color)

        selected_color = st.sidebar.color_picker(
            f"ラベル '{label}' の色",
            value=current_color,
            key=f"color_{label}"
        )

        st.session_state.label_colors[label] = selected_color

    # 色リセットボタン
    if st.sidebar.button("🎨 色をデフォルトにリセット"):
        st.session_state.label_colors = {}
        st.rerun()

def create_sample_selection_ui(labels):
    """サンプル選択UIを作成"""
    if labels is None:
        return

    unique_labels = np.unique(labels)

    # ラベルが変更された場合は選択をリセット
    if st.session_state.current_labels != list(unique_labels):
        st.session_state.current_labels = list(unique_labels)
        st.session_state.selected_labels = list(unique_labels)
        st.session_state.show_all_samples = True

    st.sidebar.markdown("### 📊 表示サンプル選択")

    # 全サンプル表示の切り替え
    show_all = st.sidebar.checkbox(
        "すべてのサンプルを表示",
        value=st.session_state.show_all_samples,
        key="show_all_samples_checkbox",
        help="チェックを外すと特定のラベルのみを選択できます"
    )

    st.session_state.show_all_samples = show_all

    if not show_all:
        # ラベル選択（マルチセレクト）
        selected_labels = st.sidebar.multiselect(
            "表示するラベルを選択",
            options=unique_labels,
            default=st.session_state.selected_labels if st.session_state.selected_labels else unique_labels,
            key="label_multiselect",
            help="グラフに表示するラベルを選択してください"
        )

        st.session_state.selected_labels = selected_labels

        # 選択されたラベルの情報を表示
        if selected_labels:
            st.sidebar.success(f"✅ {len(selected_labels)}個のラベルを選択中")
            for label in selected_labels:
                count = np.sum(labels == label)
                st.sidebar.write(f"  • {label}: {count}サンプル")
        else:
            st.sidebar.warning("⚠️ 表示するラベルが選択されていません")
    else:
        st.session_state.selected_labels = list(unique_labels)
        st.sidebar.info("✅ すべてのサンプルを表示中")

def get_filtered_data(features, labels):
    """選択されたサンプルのみを返す"""
    if labels is None or st.session_state.show_all_samples:
        return features, labels

    if not st.session_state.selected_labels:
        return features, labels

    # 選択されたラベルのサンプルのみを抽出
    mask = np.isin(labels, st.session_state.selected_labels)
    filtered_features = features[mask]
    filtered_labels = labels[mask] if labels is not None else None

    return filtered_features, filtered_labels

def apply_white_board_correction(features, white_board_spectrum):
    """白色板スペクトルによる補正を適用"""
    # 白色板スペクトルの長さを特徴量数に合わせる
    if len(white_board_spectrum) != features.shape[1]:
        # 線形補間で白色板スペクトルを調整
        original_indices = np.linspace(0, len(white_board_spectrum) - 1, len(white_board_spectrum))
        new_indices = np.linspace(0, len(white_board_spectrum) - 1, features.shape[1])
        f = interp1d(original_indices, white_board_spectrum, kind='linear')
        adjusted_white_board = f(new_indices)
    else:
        adjusted_white_board = white_board_spectrum

    # 白色板補正を適用（各特徴量を白色板スペクトルで除算）
    corrected_features = features / adjusted_white_board
    return corrected_features

def create_mean_spectrum_plot(features, labels, feature_columns, label_col):
    """ラベルごとの平均スペクトルグラフを作成"""
    if labels is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    label_colors = get_label_colors(labels)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        mean_spectrum = np.mean(features[mask], axis=0)
        std_spectrum = np.std(features[mask], axis=0)

        # ユーザー設定の色またはデフォルト色を使用
        color = label_colors.get(label, plt.cm.tab10(i))

        # 平均スペクトルをプロット（太線）
        ax.plot(
            range(len(feature_columns)), mean_spectrum,
            marker=markers[i % len(markers)], linewidth=2.5, markersize=6,
            color=color, label=f'{label}'
        )

        # 標準偏差の範囲を塗りつぶし
        ax.fill_between(
            range(len(feature_columns)),
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            alpha=0.15, color=color
        )

    ax.set_xlabel('波長 (nm)', fontsize=24)
    ax.set_ylabel('強度 (arb. unit)', fontsize=24)

    # X軸のラベルを設定
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.2, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1.0)
    ax.minorticks_on()
    ax.legend(frameon=True, loc='best', fontsize=16)

    return fig

def create_corrected_mean_spectrum_plot(features, labels, feature_columns, label_col, white_board_spectrum):
    """白色板補正後のラベルごとの平均スペクトルグラフを作成"""
    if labels is None:
        return None

    # 白色板補正を適用
    corrected_features = apply_white_board_correction(features, white_board_spectrum)

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    label_colors = get_label_colors(labels)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        mean_spectrum = np.mean(corrected_features[mask], axis=0)
        std_spectrum = np.std(corrected_features[mask], axis=0)

        # ユーザー設定の色またはデフォルト色を使用
        color = label_colors.get(label, plt.cm.tab10(i))

        # 平均スペクトルをプロット（太線）
        ax.plot(
            range(len(feature_columns)), mean_spectrum,
            marker=markers[i % len(markers)], linewidth=2.5, markersize=6,
            color=color, label=f'{label}'
        )

        # 標準偏差の範囲を塗りつぶし
        ax.fill_between(
            range(len(feature_columns)),
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            alpha=0.15, color=color
        )

    ax.set_xlabel('波長 (nm)', fontsize=24)
    ax.set_ylabel('強度 (arb. unit)', fontsize=24)

    # X軸のラベルを設定
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.2, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1.0)
    ax.minorticks_on()
    ax.legend(frameon=True, loc='best', fontsize=16)

    return fig

def perform_pca_analysis(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 第1〜第3主成分まで算出
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_result, pca, explained_variance_ratio

def create_pca_scatter_plot(pca_result, labels, explained_variance_ratio, dims=(0, 1)):
    fig, ax = plt.subplots(figsize=(10, 8))

    x_idx, y_idx = dims
    if labels is not None:
        unique_labels = np.unique(labels)
        label_colors = get_label_colors(labels)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for i, label in enumerate(unique_labels):
            mask = labels == label
            # ユーザー設定の色またはデフォルト色を使用
            color = label_colors.get(label, plt.cm.tab10(i))

            ax.scatter(
                pca_result[mask, x_idx], pca_result[mask, y_idx],
                c=[color], marker=markers[i % len(markers)], s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5, label=f'{label}'
            )
    else:
        ax.scatter(
            pca_result[:, x_idx], pca_result[:, y_idx],
            c='tab:blue', marker='o', s=100, alpha=0.9,
            edgecolors='black', linewidth=0.5
        )

    ax.set_xlabel(f'第{x_idx+1}主成分 (説明率{explained_variance_ratio[x_idx]:.1%})', fontsize=24)
    ax.set_ylabel(f'第{y_idx+1}主成分 (説明率{explained_variance_ratio[y_idx]:.1%})', fontsize=24)

    # 体裁（論文体裁）
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.2, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1.0)
    ax.minorticks_on()
    # 上下右の目盛は出すが、数字は表示しない
    # タイトルは不要

    if labels is not None:
        ax.legend(frameon=True, loc='best', fontsize=24)

    return fig

def create_loading_plot(pca, feature_columns, pc_index: int = 0):
    fig, ax = plt.subplots(figsize=(12, 6))

    pc1_loadings = pca.components_[pc_index]

    try:
        feature_importance = list(zip(feature_columns, pc1_loadings))
        feature_importance.sort(key=lambda x: int(x[0]))
    except:
        feature_importance = list(zip(feature_columns, pc1_loadings))

    sorted_features = [item[0] for item in feature_importance]
    sorted_loadings = [item[1] for item in feature_importance]

    ax.plot(
        range(len(sorted_features)), sorted_loadings,
        marker='o', linewidth=2.5, markersize=6, color='tab:blue'
    )

    for i, loading in enumerate(sorted_loadings):
        if loading < 0:
            ax.plot(i, loading, 'o', markersize=6, color='red')

    ax.set_xlabel('波長 (nm)', fontsize=24)
    ax.set_ylabel(f'第{pc_index+1}主成分負荷率', fontsize=24)

    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    mean_loading = np.mean(sorted_loadings)
    ax.axhline(y=mean_loading, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

    max_abs_loading = max(abs(min(sorted_loadings)), abs(max(sorted_loadings)))
    y_margin = max_abs_loading * 0.2
    ax.set_ylim(mean_loading - max_abs_loading - y_margin, mean_loading + max_abs_loading + y_margin)

    for i, loading in enumerate(sorted_loadings):
        ax.text(
            i, loading + (0.005 if loading >= 0 else -0.005),
            f'{loading:.3f}', ha='center', va='bottom' if loading >= 0 else 'top',
            fontsize=14
        )

    # 体裁（論文体裁）
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.2, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1.0)
    ax.minorticks_on()
    # 副軸（上・右）
    ax.secondary_xaxis('top')
    ax.secondary_yaxis('right')
    # タイトルは不要

    return fig

def create_pdf_report(report_data):
    """PDFレポートを作成"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
    story = []

    # スタイルの設定
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        spaceAfter=40,
        alignment=1,  # 中央揃え
        fontName='HeiseiKakuGo-W5',
        textColor=colors.HexColor('#2c3e50')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=15,
        spaceBefore=25,
        fontName='HeiseiKakuGo-W5',
        textColor=colors.HexColor('#34495e'),
        borderWidth=1,
        borderColor=colors.HexColor('#3498db'),
        borderPadding=8,
        backColor=colors.HexColor('#ecf0f1')
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        fontName='HeiseiKakuGo-W5',
        textColor=colors.HexColor('#2c3e50')
    )
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        fontName='HeiseiKakuGo-W5',
        textColor=colors.HexColor('#7f8c8d'),
        leftIndent=20
    )

    # タイトル
    story.append(Paragraph(report_data['title'], title_style))
    story.append(Spacer(1, 30))

    # 基本情報（カード形式）
    info_data = []
    if report_data['author']:
        info_data.append(['作成者', report_data['author']])
    if report_data['date']:
        info_data.append(['作成日', report_data['date']])

    if info_data:
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'HeiseiKakuGo-W5'),
            ('FONTNAME', (1, 0), (1, -1), 'HeiseiKakuGo-W5'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 25))

    # データ概要
    story.append(Paragraph("📊 データ概要", heading_style))

    overview_data = [
        ['項目', '値'],
        ['データ形状', report_data['data_shape']],
        ['特徴量数', str(report_data['feature_count'])],
    ]
    if report_data['label_info']:
        overview_data.append(['ラベル情報', report_data['label_info']])
    if 'sample_selection' in report_data:
        overview_data.append(['表示サンプル', report_data['sample_selection']])

    overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'HeiseiKakuGo-W5'),
        ('FONTNAME', (0, 1), (-1, -1), 'HeiseiKakuGo-W5'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 25))

    # PCA結果（見出し順の整理）
    story.append(Paragraph("🔬 PCA分析結果の要約", heading_style))

    # 使用データの情報を追加
    if 'pca_data_type' in report_data:
        story.append(Paragraph(f"<b>📊 使用データ:</b> {report_data['pca_data_type']}", normal_style))
        story.append(Spacer(1, 15))

    pca_data = [
        ['主成分', '寄与率', '累積寄与率'],
        ['第1主成分', f"{report_data['pc1_variance']:.1%}", f"{report_data['pc1_variance']:.1%}"],
        ['第2主成分', f"{report_data['pc2_variance']:.1%}", f"{report_data['pc1_variance'] + report_data['pc2_variance']:.1%}"],
    ]

    if 'pc3_variance' in report_data and report_data['pc3_variance'] is not None:
        pca_data.append(['第3主成分', f"{report_data['pc3_variance']:.1%}", f"{report_data['total_variance']:.1%}"])

    pca_table = Table(pca_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    pca_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'HeiseiKakuGo-W5'),
        ('FONTNAME', (0, 1), (-1, -1), 'HeiseiKakuGo-W5'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(pca_table)
    story.append(Spacer(1, 25))

    # グラフと考察
    if report_data['mean_spectrum_fig']:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph("📈 ラベルごとの平均スペクトル（生データ）", heading_style))
        story.append(Image(report_data['mean_spectrum_fig'], width=6*inch, height=4*inch))
        if report_data['mean_spectrum_notes']:
            story.append(Paragraph(f"<b>💭 考察:</b> {report_data['mean_spectrum_notes']}", normal_style))
        story.append(Spacer(1, 25))

    if report_data.get('corrected_spectrum_fig'):
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph("📈 ラベルごとの平均スペクトル（白色板補正後）", heading_style))
        story.append(Image(report_data['corrected_spectrum_fig'], width=6*inch, height=4*inch))
        if report_data.get('corrected_spectrum_notes'):
            story.append(Paragraph(f"<b>💭 考察:</b> {report_data['corrected_spectrum_notes']}", normal_style))
        story.append(Spacer(1, 25))

    # PCA散布図の見出しに使用データの情報を追加
    pca_data_type_text = ""
    if 'pca_data_type' in report_data:
        pca_data_type_text = f" ({report_data['pca_data_type']})"

    story.append(PageBreak())  # 新しいページに開始
    story.append(Paragraph(f"📊 PCA散布図 (PC1 vs PC2){pca_data_type_text}", heading_style))
    story.append(Image(report_data['pca_scatter_fig'], width=6*inch, height=4*inch))
    if report_data['pca_scatter_notes']:
        story.append(Paragraph(f"<b>💭 考察:</b> {report_data['pca_scatter_notes']}", normal_style))
    story.append(Spacer(1, 25))

    if 'pca_scatter_fig_pc13' in report_data and report_data['pca_scatter_fig_pc13'] is not None:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph(f"📊 PCA散布図 (PC1 vs PC3){pca_data_type_text}", heading_style))
        story.append(Image(report_data['pca_scatter_fig_pc13'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 25))

    if 'pca_scatter_fig_pc23' in report_data and report_data['pca_scatter_fig_pc23'] is not None:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph(f"📊 PCA散布図 (PC2 vs PC3){pca_data_type_text}", heading_style))
        story.append(Image(report_data['pca_scatter_fig_pc23'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 25))

    story.append(PageBreak())  # 新しいページに開始
    story.append(Paragraph(f"📉 負荷率グラフ (PC1){pca_data_type_text}", heading_style))
    story.append(Image(report_data['loading_fig'], width=6*inch, height=3*inch))
    if report_data['loading_notes']:
        story.append(Paragraph(f"<b>💭 考察:</b> {report_data['loading_notes']}", normal_style))
    story.append(Spacer(1, 25))

    if 'loading_fig_pc2' in report_data and report_data['loading_fig_pc2'] is not None:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph(f"📉 負荷率グラフ (PC2){pca_data_type_text}", heading_style))
        story.append(Image(report_data['loading_fig_pc2'], width=6*inch, height=3*inch))
        story.append(Spacer(1, 25))

    if 'loading_fig_pc3' in report_data and report_data['loading_fig_pc3'] is not None:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph(f"📉 負荷率グラフ (PC3){pca_data_type_text}", heading_style))
        story.append(Image(report_data['loading_fig_pc3'], width=6*inch, height=3*inch))
        story.append(Spacer(1, 25))

    # 総合考察
    if report_data['overall_conclusion']:
        story.append(PageBreak())  # 新しいページに開始
        story.append(Paragraph("💡 総合考察", heading_style))
        story.append(Paragraph(report_data['overall_conclusion'], normal_style))
        story.append(Spacer(1, 25))

    # 特徴量重要度表
    story.append(PageBreak())  # 新しいページに開始
    story.append(Paragraph(f"🎯 特徴量重要度（第1主成分）{pca_data_type_text}", heading_style))
    importance_data = [['特徴量', '負荷率', '絶対値']]
    for feature, loading in report_data['feature_importance'][:10]:
        importance_data.append([feature, f"{loading:.3f}", f"{abs(loading):.3f}"])

    importance_table = Table(importance_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'HeiseiKakuGo-W5'),
        ('FONTNAME', (0, 1), (-1, -1), 'HeiseiKakuGo-W5'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(importance_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

def save_figure_as_image(fig):
    """matplotlibの図を画像ファイルとして保存"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    return img_buffer

def main():
    # サイドバーのスタイリング
    st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
    }
    .sidebar-header {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # レポート情報入力セクション
    st.sidebar.markdown('<div class="sidebar-header">📝 レポート情報</div>', unsafe_allow_html=True)

    st.session_state.report_title = st.sidebar.text_input(
        "レポートタイトル",
        value=st.session_state.report_title,
        help="レポートのタイトルを入力してください"
    )

    st.session_state.author_name = st.sidebar.text_input(
        "作成者名",
        value=st.session_state.author_name,
        help="レポートの作成者名を入力してください"
    )

    selected_date = st.sidebar.date_input(
        "分析日",
        value=(st.session_state.analysis_date if st.session_state.analysis_date is not None else pd.Timestamp.now().date()),
        help="分析を実行した日付を選択してください"
    )
    st.session_state.analysis_date = selected_date

    st.sidebar.markdown("---")

    st.sidebar.markdown('<div class="sidebar-header">📁 ファイルアップロード</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "CSVファイルを選択してください",
        type=['csv'],
        help="Label列（オプション）と数値データを含むCSVファイルをアップロードしてください"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ ファイル '{uploaded_file.name}' を正常に読み込みました")

            # データ概要をカード形式で表示
            st.markdown('<div class="section-header">📊 データ概要</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">📈 行数</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{len(df)}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">📋 列数</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{len(df.columns)}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">🔢 数値列</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{numeric_cols}</p>
                </div>
                """, unsafe_allow_html=True)

            features, labels, feature_columns, label_col = load_and_prepare_data(df)

            if len(feature_columns) < 2:
                st.error("❌ 数値列が2つ以上必要です。")
                return

            # 色選択UIを表示
            create_color_picker_ui(labels)

            # サンプル選択UIを表示
            create_sample_selection_ui(labels)

            st.subheader("📋 データ詳細")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**データ形状:**", features.shape)
                st.write("**特徴量数:**", len(feature_columns))
                if labels is not None:
                    st.write("**ラベル列:**", label_col)
                    st.write("**ラベル種類:**", list(np.unique(labels)))

            with col2:
                st.write("**特徴量名:**")
                st.write(feature_columns)

            # ラベルごとの平均スペクトルグラフを表示
            if labels is not None:
                st.subheader("📈 ラベルごとの平均スペクトル")

                # タブで生データと補正データを分ける
                tab_original, tab_corrected = st.tabs(["生データ", "白色板補正後"])

                with tab_original:
                    # 選択されたサンプルのみを使用
                    filtered_features, filtered_labels = get_filtered_data(features, labels)
                    mean_spectrum_fig = create_mean_spectrum_plot(filtered_features, filtered_labels, feature_columns, label_col)
                    if mean_spectrum_fig:
                        st.pyplot(mean_spectrum_fig)

                        # 考察用テキストボックス
                        st.markdown("**📝 このグラフに関する考察や結果を記入してください:**")
                        st.session_state.mean_spectrum_notes = st.text_area(
                            "平均スペクトルグラフの考察",
                            value=st.session_state.mean_spectrum_notes,
                            height=100,
                            placeholder="このグラフから分かったこと、クラス間の違い、特徴的なパターンなどを記入してください..."
                        )

                with tab_corrected:
                    # 選択されたサンプルのみを使用
                    filtered_features, filtered_labels = get_filtered_data(features, labels)
                    corrected_spectrum_fig = create_corrected_mean_spectrum_plot(filtered_features, filtered_labels, feature_columns, label_col, WHITE_BOARD_SPECTRUM)
                    if corrected_spectrum_fig:
                        st.pyplot(corrected_spectrum_fig)

                        # 考察用テキストボックス
                        st.markdown("**📝 白色板補正後のグラフに関する考察や結果を記入してください:**")
                        st.session_state.corrected_spectrum_notes = st.text_area(
                            "白色板補正後スペクトルグラフの考察",
                            value=st.session_state.corrected_spectrum_notes,
                            height=100,
                            placeholder="補正前後での違い、クラス間の違いの変化、特徴的なパターンなどを記入してください..."
                        )
                st.markdown("---")

            st.subheader("🔬 PCA分析実行")

            # PCA分析用データ選択
            st.markdown("**📊 PCA分析に使用するデータを選択してください：**")
            col1, col2 = st.columns(2)

            with col1:
                use_original = st.radio(
                    "データ選択",
                    ["生データ", "白色板補正後データ"],
                    key="pca_data_selection",
                    help="PCA分析に使用するデータを選択してください"
                )
                st.session_state.use_corrected_data_for_pca = (use_original == "白色板補正後データ")

            with col2:
                if st.session_state.use_corrected_data_for_pca:
                    st.info("✅ 白色板補正後データを使用してPCA分析を実行します")
                else:
                    st.info("✅ 生データを使用してPCA分析を実行します")

            # PCA分析に使用するデータを決定（選択されたサンプルのみ）
            filtered_features, filtered_labels = get_filtered_data(features, labels)
            pca_features = filtered_features
            if st.session_state.use_corrected_data_for_pca:
                pca_features = apply_white_board_correction(filtered_features, WHITE_BOARD_SPECTRUM)

            with st.spinner("PCA分析を実行中..."):
                pca_result, pca, explained_variance_ratio = perform_pca_analysis(pca_features)

            st.success("✅ PCA分析が完了しました")

            # 使用したデータの情報を表示
            data_type = "白色板補正後データ" if st.session_state.use_corrected_data_for_pca else "生データ"
            st.info(f"📊 使用データ: {data_type}")

            # PCA分析結果をカード形式で表示
            st.markdown('<div class="section-header">🔬 PCA分析結果</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">PC1 寄与率</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold; color: #e74c3c;">{explained_variance_ratio[0]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">PC2 寄与率</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold; color: #e74c3c;">{explained_variance_ratio[1]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                pc3_var = explained_variance_ratio[2] if len(explained_variance_ratio) > 2 else 0
                pc3_display = f"{pc3_var:.1%}" if len(explained_variance_ratio) > 2 else "-"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">PC3 寄与率</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold; color: #e74c3c;">{pc3_display}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                total_var = sum(explained_variance_ratio[:3]) if len(explained_variance_ratio) >= 3 else sum(explained_variance_ratio[:2])
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; color: #667eea;">累積寄与率</h3>
                    <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold; color: #27ae60;">{total_var:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("📊 分析結果グラフ")

            # すべての図を事前生成（PDF出力に備える）- 選択されたサンプルのみを使用
            fig_pc12 = create_pca_scatter_plot(pca_result, filtered_labels, explained_variance_ratio, dims=(0, 1))
            fig_pc13 = create_pca_scatter_plot(pca_result, filtered_labels, explained_variance_ratio, dims=(0, 2)) if pca_result.shape[1] >= 3 else None
            fig_pc23 = create_pca_scatter_plot(pca_result, filtered_labels, explained_variance_ratio, dims=(1, 2)) if pca_result.shape[1] >= 3 else None
            loading_fig = create_loading_plot(pca, feature_columns)

            tab1, tab2, tab3, tab4 = st.tabs(["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3", "負荷率グラフ（PC1/PC2/PC3）"])

            with tab1:
                st.pyplot(fig_pc12)

                # PCA散布図の考察用テキストボックス
                st.markdown("**📝 このグラフに関する考察や結果を記入してください:**")
                st.session_state.pca_scatter_notes = st.text_area(
                    "PCA散布図の考察",
                    value=st.session_state.pca_scatter_notes,
                    height=100,
                    placeholder="クラスターの形成、サンプル間の関係性、外れ値の有無などを記入してください..."
                )

            with tab2:
                if fig_pc13 is not None:
                    st.pyplot(fig_pc13)
                else:
                    st.info("PC3が存在しないため、このグラフは表示できません。")

                st.markdown("**📝 このグラフに関する考察や結果を記入してください:**")
                st.session_state.pca_scatter_notes = st.text_area(
                    "PCA散布図の考察 (PC1 vs PC3)",
                    value=st.session_state.pca_scatter_notes,
                    height=100,
                    placeholder="クラスターの形成、サンプル間の関係性、外れ値の有無などを記入してください..."
                )

            with tab3:
                if fig_pc23 is not None:
                    st.pyplot(fig_pc23)
                else:
                    st.info("PC3が存在しないため、このグラフは表示できません。")

                st.markdown("**📝 このグラフに関する考察や結果を記入してください:**")
                st.session_state.pca_scatter_notes = st.text_area(
                    "PCA散布図の考察 (PC2 vs PC3)",
                    value=st.session_state.pca_scatter_notes,
                    height=100,
                    placeholder="クラスターの形成、サンプル間の関係性、外れ値の有無などを記入してください..."
                )

            with tab4:
                st.markdown("#### PC1 負荷率")
                st.pyplot(create_loading_plot(pca, feature_columns, pc_index=0))
                st.markdown("#### PC2 負荷率")
                st.pyplot(create_loading_plot(pca, feature_columns, pc_index=1))
                if pca.n_components_ >= 3:
                    st.markdown("#### PC3 負荷率")
                    st.pyplot(create_loading_plot(pca, feature_columns, pc_index=2))

                # 負荷率グラフの考察用テキストボックス
                st.markdown("**📝 このグラフに関する考察や結果を記入してください:**")
                st.session_state.loading_notes = st.text_area(
                    "負荷率グラフの考察",
                    value=st.session_state.loading_notes,
                    height=100,
                    placeholder="重要な特徴量、正負の相関、特徴量のグループ化などを記入してください..."
                )

            st.subheader("📈 詳細分析結果")

            pc1_loadings = pca.components_[0]
            feature_importance = list(zip(feature_columns, pc1_loadings))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            st.write("**第1主成分への寄与度上位10特徴量:**")
            importance_df = pd.DataFrame(feature_importance[:10], columns=['特徴量', '負荷率'])
            importance_df['絶対値'] = importance_df['負荷率'].abs()
            importance_df = importance_df.sort_values('絶対値', ascending=False)
            st.dataframe(importance_df, use_container_width=True)

            # 総合考察セクション
            st.subheader("📝 総合考察")
            st.markdown("**分析全体を通じての総合的な考察や結論を記入してください:**")
            st.session_state.overall_conclusion = st.text_area(
                "総合考察",
                value=st.session_state.overall_conclusion,
                height=150,
                placeholder="PCA分析の結果から得られた全体的な知見、実用的な意味、今後の研究方向などを記入してください..."
            )

            st.subheader("💾 結果のダウンロード")

            col1, col2 = st.columns(2)

            with col1:
                # PC1-3 を出力
                pc_cols = ['PC1', 'PC2', 'PC3'] if pca_result.shape[1] >= 3 else ['PC1', 'PC2']
                pca_df = pd.DataFrame(pca_result[:, :len(pc_cols)], columns=pc_cols)
                if labels is not None:
                    pca_df['Label'] = labels

                csv_buffer = io.StringIO()
                pca_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="PCA結果をダウンロード (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="pca_results.csv",
                    mime="text/csv"
                )

            with col2:
                # PDFレポート作成ボタン
                if st.button("📄 PDFレポートを作成", type="primary"):
                    with st.spinner("PDFレポートを作成中..."):
                        try:
                            # 図を画像として保存
                            mean_spectrum_img = save_figure_as_image(mean_spectrum_fig) if labels is not None else None
                            corrected_spectrum_img = save_figure_as_image(corrected_spectrum_fig) if labels is not None else None
                            pca_scatter_img = save_figure_as_image(fig_pc12)
                            pca_scatter_img_pc13 = save_figure_as_image(fig_pc13) if fig_pc13 is not None else None
                            pca_scatter_img_pc23 = save_figure_as_image(fig_pc23) if fig_pc23 is not None else None
                            loading_img_pc1 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=0))
                            loading_img_pc2 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=1))
                            loading_img_pc3 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=2)) if pca.n_components_ >= 3 else None

                            # 選択されたサンプルの情報を準備
                            sample_info = ""
                            if labels is not None:
                                if st.session_state.show_all_samples:
                                    sample_info = f"すべてのサンプル ({len(features)}サンプル)"
                                else:
                                    selected_labels = st.session_state.selected_labels
                                    if selected_labels:
                                        total_selected_samples = sum(np.sum(labels == label) for label in selected_labels)
                                        sample_info = f"選択されたラベル: {selected_labels} ({total_selected_samples}サンプル)"
                                    else:
                                        sample_info = "サンプルが選択されていません"
                            else:
                                sample_info = f"すべてのサンプル ({len(features)}サンプル)"

                            # レポートデータの準備
                            report_data = {
                                'title': st.session_state.report_title,
                                'author': st.session_state.author_name,
                                'date': (st.session_state.analysis_date.strftime("%Y年%m月%d日") if st.session_state.analysis_date else ""),
                                'data_shape': f"{features.shape[0]}行 × {features.shape[1]}列",
                                'feature_count': len(feature_columns),
                                'label_info': f"{label_col}: {list(np.unique(labels))}" if labels is not None else "なし",
                                'sample_selection': sample_info,
                                'pca_data_type': "白色板補正後データ" if st.session_state.use_corrected_data_for_pca else "生データ",
                                'pc1_variance': explained_variance_ratio[0],
                                'pc2_variance': explained_variance_ratio[1],
                                'total_variance': sum(explained_variance_ratio[:3]) if len(explained_variance_ratio) >= 3 else sum(explained_variance_ratio[:2]),
                                'pc3_variance': (explained_variance_ratio[2] if len(explained_variance_ratio) >= 3 else None),
                                'mean_spectrum_fig': mean_spectrum_img,
                                'corrected_spectrum_fig': corrected_spectrum_img,
                                'pca_scatter_fig': pca_scatter_img,
                                'pca_scatter_fig_pc13': pca_scatter_img_pc13,
                                'pca_scatter_fig_pc23': pca_scatter_img_pc23,
                                'loading_fig': loading_img_pc1,
                                'loading_fig_pc2': loading_img_pc2,
                                'loading_fig_pc3': loading_img_pc3,
                                'mean_spectrum_notes': st.session_state.mean_spectrum_notes,
                                'corrected_spectrum_notes': st.session_state.corrected_spectrum_notes,
                                'pca_scatter_notes': st.session_state.pca_scatter_notes,
                                'loading_notes': st.session_state.loading_notes,
                                'overall_conclusion': st.session_state.overall_conclusion,
                                'feature_importance': feature_importance[:10]
                            }

                            # PDFレポート作成
                            pdf_buffer = create_pdf_report(report_data)

                            # PDFダウンロードボタン
                            st.download_button(
                                label="📥 PDFレポートをダウンロード",
                                data=pdf_buffer.getvalue(),
                                file_name=f"PCA_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )

                            st.success("✅ PDFレポートが作成されました！")

                        except Exception as e:
                            st.error(f"❌ PDFレポートの作成中にエラーが発生しました: {str(e)}")

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.write("ファイル形式を確認してください。")

    else:
        st.info("👆 左側のサイドバーからCSVファイルをアップロードしてください")

        st.markdown("""
        ### 📋 対応ファイル形式

        **CSVファイルの要件:**
        - 数値データを含む列が2つ以上必要です
        - ラベル列（オプション）: 'Label', 'Class', 'Category', 'Group' などの名前

        **例:**
        ```csv
        Label,feature1,feature2,feature3
        A,1.2,3.4,5.6
        B,2.1,4.3,6.5
        C,1.8,3.9,5.2
        ```
        """)

        st.markdown("### 📥 サンプルデータ")
        sample_data = pd.DataFrame({
            'Label': ['A', 'A', 'B', 'B', 'C', 'C'] * 3,
            'feature1': np.random.randn(18),
            'feature2': np.random.randn(18),
            'feature3': np.random.randn(18),
            'feature4': np.random.randn(18)
        })

        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="サンプルデータをダウンロード",
            data=csv_buffer.getvalue(),
            file_name="sample_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
