import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
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

st.set_page_config(page_title="irodori 解析アプリ", layout="wide")

st.title("📊 主成分分析（PCA）レポート作成アプリケーション")
st.markdown("---")

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

def create_mean_spectrum_plot(features, labels, feature_columns, label_col):
    """ラベルごとの平均スペクトルグラフを作成"""
    if labels is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        mean_spectrum = np.mean(features[mask], axis=0)
        std_spectrum = np.std(features[mask], axis=0)

        # 平均スペクトルをプロット（太線）
        ax.plot(
            range(len(feature_columns)), mean_spectrum,
            marker=markers[i % len(markers)], linewidth=2.5, markersize=6,
            color=colors[i % len(colors)], label=f'クラス {label}'
        )

        # 標準偏差の範囲を塗りつぶし
        ax.fill_between(
            range(len(feature_columns)),
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            alpha=0.15, color=colors[i % len(colors)]
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
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                pca_result[mask, x_idx], pca_result[mask, y_idx],
                c=[colors[i % len(colors)]], marker=markers[i % len(markers)], s=100, alpha=0.9,
                edgecolors='black', linewidth=0.5, label=f'クラス {label}'
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
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []

    # スタイルの設定
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # 中央揃え
        fontName='HeiseiKakuGo-W5'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        fontName='HeiseiKakuGo-W5'
    )
    normal_style = styles['Normal']
    normal_style.fontName = 'HeiseiKakuGo-W5'

    # タイトル
    story.append(Paragraph(report_data['title'], title_style))
    story.append(Spacer(1, 20))

    # 基本情報
    if report_data['author']:
        story.append(Paragraph(f"<b>作成者:</b> {report_data['author']}", normal_style))
    if report_data['date']:
        story.append(Paragraph(f"<b>作成日:</b> {report_data['date']}", normal_style))
    story.append(Spacer(1, 20))

    # データ概要
    story.append(Paragraph("データ概要", heading_style))
    story.append(Paragraph(f"<b>データ形状:</b> {report_data['data_shape']}", normal_style))
    story.append(Paragraph(f"<b>特徴量数:</b> {report_data['feature_count']}", normal_style))
    if report_data['label_info']:
        story.append(Paragraph(f"<b>ラベル情報:</b> {report_data['label_info']}", normal_style))
    story.append(Spacer(1, 20))

    # PCA結果（見出し順の整理）
    story.append(Paragraph("PCA分析結果の要約", heading_style))
    story.append(Paragraph(f"<b>第1主成分寄与率:</b> {report_data['pc1_variance']:.1%}", normal_style))
    story.append(Paragraph(f"<b>第2主成分寄与率:</b> {report_data['pc2_variance']:.1%}", normal_style))
    if 'pc3_variance' in report_data and report_data['pc3_variance'] is not None:
        story.append(Paragraph(f"<b>第3主成分寄与率:</b> {report_data['pc3_variance']:.1%}", normal_style))
    story.append(Paragraph(f"<b>累積寄与率:</b> {report_data['total_variance']:.1%}", normal_style))
    story.append(Spacer(1, 20))

    # グラフと考察
    if report_data['mean_spectrum_fig']:
        story.append(Paragraph("ラベルごとの平均スペクトル", heading_style))
        story.append(Image(report_data['mean_spectrum_fig'], width=6*inch, height=4*inch))
        if report_data['mean_spectrum_notes']:
            story.append(Paragraph(f"<b>考察:</b> {report_data['mean_spectrum_notes']}", normal_style))
        story.append(Spacer(1, 20))

    story.append(Paragraph("PCA散布図 (PC1 vs PC2)", heading_style))
    story.append(Image(report_data['pca_scatter_fig'], width=6*inch, height=4*inch))
    if report_data['pca_scatter_notes']:
        story.append(Paragraph(f"<b>考察:</b> {report_data['pca_scatter_notes']}", normal_style))
    story.append(Spacer(1, 20))

    if 'pca_scatter_fig_pc13' in report_data and report_data['pca_scatter_fig_pc13'] is not None:
        story.append(Paragraph("PCA散布図 (PC1 vs PC3)", heading_style))
        story.append(Image(report_data['pca_scatter_fig_pc13'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 20))

    if 'pca_scatter_fig_pc23' in report_data and report_data['pca_scatter_fig_pc23'] is not None:
        story.append(Paragraph("PCA散布図 (PC2 vs PC3)", heading_style))
        story.append(Image(report_data['pca_scatter_fig_pc23'], width=6*inch, height=4*inch))
        story.append(Spacer(1, 20))

    story.append(Paragraph("負荷率グラフ (PC1)", heading_style))
    story.append(Image(report_data['loading_fig'], width=6*inch, height=3*inch))
    if report_data['loading_notes']:
        story.append(Paragraph(f"<b>考察:</b> {report_data['loading_notes']}", normal_style))
    story.append(Spacer(1, 20))

    if 'loading_fig_pc2' in report_data and report_data['loading_fig_pc2'] is not None:
        story.append(Paragraph("負荷率グラフ (PC2)", heading_style))
        story.append(Image(report_data['loading_fig_pc2'], width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))

    if 'loading_fig_pc3' in report_data and report_data['loading_fig_pc3'] is not None:
        story.append(Paragraph("負荷率グラフ (PC3)", heading_style))
        story.append(Image(report_data['loading_fig_pc3'], width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))

    # 総合考察
    if report_data['overall_conclusion']:
        story.append(Paragraph("総合考察", heading_style))
        story.append(Paragraph(report_data['overall_conclusion'], normal_style))
        story.append(Spacer(1, 20))

    # 特徴量重要度表
    story.append(Paragraph("特徴量重要度（第1主成分）", heading_style))
    importance_data = [['特徴量', '負荷率', '絶対値']]
    for feature, loading in report_data['feature_importance'][:10]:
        importance_data.append([feature, f"{loading:.3f}", f"{abs(loading):.3f}"])

    importance_table = Table(importance_data)
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'HeiseiKakuGo-W5')
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
    # レポート情報入力セクション
    st.sidebar.header("📝 レポート情報")
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

    st.sidebar.header("📁 ファイルアップロード")
    uploaded_file = st.sidebar.file_uploader(
        "CSVファイルを選択してください",
        type=['csv'],
        help="Label列（オプション）と数値データを含むCSVファイルをアップロードしてください"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ ファイル '{uploaded_file.name}' を正常に読み込みました")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", len(df))
            with col2:
                st.metric("列数", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("数値列", numeric_cols)

            features, labels, feature_columns, label_col = load_and_prepare_data(df)

            if len(feature_columns) < 2:
                st.error("❌ 数値列が2つ以上必要です。")
                return

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
                mean_spectrum_fig = create_mean_spectrum_plot(features, labels, feature_columns, label_col)
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
                st.markdown("---")

            st.subheader("🔬 PCA分析実行")
            with st.spinner("PCA分析を実行中..."):
                pca_result, pca, explained_variance_ratio = perform_pca_analysis(features)

            st.success("✅ PCA分析が完了しました")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("第1主成分寄与率", f"{explained_variance_ratio[0]:.1%}")
            with col2:
                st.metric("第2主成分寄与率", f"{explained_variance_ratio[1]:.1%}")
            with col3:
                if len(explained_variance_ratio) > 2:
                    st.metric("第3主成分寄与率", f"{explained_variance_ratio[2]:.1%}")
                else:
                    st.metric("第3主成分寄与率", "-")
            with col4:
                total_var = sum(explained_variance_ratio[:3]) if len(explained_variance_ratio) >= 3 else sum(explained_variance_ratio[:2])
                st.metric("累積寄与率", f"{total_var:.1%}")

            st.subheader("📊 分析結果グラフ")

            # すべての図を事前生成（PDF出力に備える）
            fig_pc12 = create_pca_scatter_plot(pca_result, labels, explained_variance_ratio, dims=(0, 1))
            fig_pc13 = create_pca_scatter_plot(pca_result, labels, explained_variance_ratio, dims=(0, 2)) if pca_result.shape[1] >= 3 else None
            fig_pc23 = create_pca_scatter_plot(pca_result, labels, explained_variance_ratio, dims=(1, 2)) if pca_result.shape[1] >= 3 else None
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
                            pca_scatter_img = save_figure_as_image(fig_pc12)
                            pca_scatter_img_pc13 = save_figure_as_image(fig_pc13) if fig_pc13 is not None else None
                            pca_scatter_img_pc23 = save_figure_as_image(fig_pc23) if fig_pc23 is not None else None
                            loading_img_pc1 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=0))
                            loading_img_pc2 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=1))
                            loading_img_pc3 = save_figure_as_image(create_loading_plot(pca, feature_columns, pc_index=2)) if pca.n_components_ >= 3 else None

                            # レポートデータの準備
                            report_data = {
                                'title': st.session_state.report_title,
                                'author': st.session_state.author_name,
                                'date': (st.session_state.analysis_date.strftime("%Y年%m月%d日") if st.session_state.analysis_date else ""),
                                'data_shape': f"{features.shape[0]}行 × {features.shape[1]}列",
                                'feature_count': len(feature_columns),
                                'label_info': f"{label_col}: {list(np.unique(labels))}" if labels is not None else "なし",
                                'pc1_variance': explained_variance_ratio[0],
                                'pc2_variance': explained_variance_ratio[1],
                                'total_variance': sum(explained_variance_ratio[:3]) if len(explained_variance_ratio) >= 3 else sum(explained_variance_ratio[:2]),
                                'pc3_variance': (explained_variance_ratio[2] if len(explained_variance_ratio) >= 3 else None),
                                'mean_spectrum_fig': mean_spectrum_img,
                                'pca_scatter_fig': pca_scatter_img,
                                'pca_scatter_fig_pc13': pca_scatter_img_pc13,
                                'pca_scatter_fig_pc23': pca_scatter_img_pc23,
                                'loading_fig': loading_img_pc1,
                                'loading_fig_pc2': loading_img_pc2,
                                'loading_fig_pc3': loading_img_pc3,
                                'mean_spectrum_notes': st.session_state.mean_spectrum_notes,
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
