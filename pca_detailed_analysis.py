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

def perform_pca_analysis(features, n_components=None):
    """PCA分析を実行"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if n_components is None:
        n_components = min(features.shape)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    return pca_result, pca, explained_variance_ratio, cumulative_variance_ratio

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

    ax.set_xlabel(f'第1主成分 ({explained_variance_ratio[0]:.1%} 分散)', fontsize=14)
    ax.set_ylabel(f'第2主成分 ({explained_variance_ratio[1]:.1%} 分散)', fontsize=14)
    ax.set_title('主成分分析（PCA）結果', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, loc='upper right', fontsize=12, labels=['クラス A', 'クラス B', 'クラス C'])

    plt.tight_layout()
    plt.savefig('pca_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

def create_loading_plot(pca, feature_columns):
    """第1主成分の負荷率プロットを作成"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 第1主成分の負荷率を取得
    pc1_loadings = pca.components_[0]

        # 波長の数値順でソート
    feature_importance = list(zip(feature_columns, pc1_loadings))
    feature_importance.sort(key=lambda x: int(x[0]))  # 波長の数値でソート

    # ソートされた特徴量名と負荷率を取得
    sorted_features = [item[0] for item in feature_importance]
    sorted_loadings = [item[1] for item in feature_importance]

        # 折れ線グラフを作成
    line = ax.plot(range(len(sorted_features)), sorted_loadings,
                   marker='o', linewidth=2, markersize=6, color='blue')

    # 負の値は赤色で表示
    for i, loading in enumerate(sorted_loadings):
        if loading < 0:
            ax.plot(i, loading, 'o', markersize=6, color='red')

    # 横軸のラベルを設定
    ax.set_xlabel('特徴量（波長）', fontsize=12)
    ax.set_ylabel('第1主成分負荷率', fontsize=12)
    ax.set_title('第1主成分の負荷率', fontsize=14, fontweight='bold')

    # 横軸のラベルを回転して表示
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')

    # グリッドと平均値ライン
    ax.grid(True, alpha=0.3, axis='y')
    mean_loading = np.mean(sorted_loadings)
    ax.axhline(y=mean_loading, color='black', linestyle='-', linewidth=0.8, alpha=0.7)

    # 縦軸の範囲を調整（平均値を中心に）
    max_abs_loading = max(abs(min(sorted_loadings)), abs(max(sorted_loadings)))
    y_margin = max_abs_loading * 0.2  # 20%のマージン
    ax.set_ylim(mean_loading - max_abs_loading - y_margin, mean_loading + max_abs_loading + y_margin)

        # 値のラベルを追加
    for i, loading in enumerate(sorted_loadings):
        ax.text(i, loading + (0.005 if loading >= 0 else -0.005),
                f'{loading:.3f}', ha='center', va='bottom' if loading >= 0 else 'top',
                fontsize=9)

    plt.tight_layout()
    plt.savefig('loading_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

def print_pca_summary(pca, feature_columns, explained_variance_ratio):
    """PCA分析の結果を表示"""
    print("=== PCA分析結果サマリー ===")
    print(f"第1・第2主成分による総分散説明率: {sum(explained_variance_ratio[:2]):.3f} ({sum(explained_variance_ratio[:2])*100:.1f}%)")
    print(f"第1主成分分散: {explained_variance_ratio[0]:.3f} ({explained_variance_ratio[0]*100:.1f}%)")
    print(f"第2主成分分散: {explained_variance_ratio[1]:.3f} ({explained_variance_ratio[1]*100:.1f}%)")

    print("\n第1主成分への寄与度上位5特徴量:")
    pc1_loadings = pca.components_[0]
    feature_importance = list(zip(feature_columns, pc1_loadings))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (feature, loading) in enumerate(feature_importance[:5]):
        print(f"  {feature}: {loading:.3f}")

def main():
    """メイン関数"""
    print("データを読み込んでいます...")
    features, labels, feature_columns = load_and_prepare_data('sample.csv')

    print(f"データ形状: {features.shape}")
    print(f"特徴量数: {len(feature_columns)}")
    print(f"サンプル数: {len(labels)}")
    print(f"ラベル: {np.unique(labels)}")

    print("\nPCA分析を実行しています...")
    pca_result, pca, explained_variance_ratio, cumulative_variance_ratio = perform_pca_analysis(features)

    print_pca_summary(pca, feature_columns, explained_variance_ratio)

    print("\nプロットを作成しています...")
    create_publication_plot(pca_result, labels, explained_variance_ratio)
    create_loading_plot(pca, feature_columns)

    print("\n分析完了！")
    print("生成されたファイル:")
    print("- pca_plot.pdf: 主成分分析散布図")
    print("- loading_plot.pdf: 第1主成分負荷率グラフ")

if __name__ == "__main__":
    main()
