import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 論文用の設定
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

def load_and_prepare_data(csv_file):
    """CSVファイルを読み込んでデータを準備"""
    df = pd.read_csv(csv_file)
    labels = df['Label'].values
    feature_columns = [col for col in df.columns if col not in ['Label', 'timestamp']]
    features = df[feature_columns].values
    return features, labels, feature_columns

def perform_pca_analysis(features, n_components=2):
    """PCA分析を実行"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    return pca_result, pca, explained_variance_ratio

def create_publication_plot(pca_result, labels, explained_variance_ratio):
    """論文品質のPCAプロットを作成"""
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                  c=colors[label], marker=markers[label], s=60, alpha=0.8,
                  edgecolors='black', linewidth=0.5, label=f'Class {label}')

    ax.set_xlabel(f'第1主成分 (説明率{explained_variance_ratio[0]:.1%} )', fontsize=14)
    ax.set_ylabel(f'第2主成分 (説明率{explained_variance_ratio[1]:.1%} )', fontsize=14)
    ax.set_title('主成分分析（PCA）結果', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, loc='upper right', fontsize=12, labels=['クラス A', 'クラス B', 'クラス C'])

    plt.tight_layout()
    plt.savefig('pca_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

def main():
    """メイン関数"""
    print("データを読み込んでいます...")
    features, labels, feature_columns = load_and_prepare_data('sample.csv')

    print(f"データ形状: {features.shape}")
    print(f"ラベル: {np.unique(labels)}")

    print("PCA分析を実行しています...")
    pca_result, pca, explained_variance_ratio = perform_pca_analysis(features)

    print(f"第1主成分寄与率: {explained_variance_ratio[0]:.3f} ({explained_variance_ratio[0]*100:.1f}%)")
    print(f"第2主成分寄与率: {explained_variance_ratio[1]:.3f} ({explained_variance_ratio[1]*100:.1f}%)")

    print("プロットを作成しています...")
    create_publication_plot(pca_result, labels, explained_variance_ratio)

    print("分析完了！pca_plot.pdfが生成されました。")

if __name__ == "__main__":
    main()
