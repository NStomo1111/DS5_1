"""
データサイエンス基礎講座：データフォーマットと可視化の実践
Complete Analysis Code

このスクリプトは講義で扱ったすべての分析手法を含んでいます：
1. データ読み込みと基本確認
2. データフォーマット変換（ワイド型⇔ロング型）
3. 可視化（棒グラフ、相関行列、PCA、散布図行列）
4. 統計的検定
"""

# ============================================================================
# 1. ライブラリのインポート
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 日本語フォントの設定（文字化け防止）
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("データサイエンス基礎講座：データフォーマットと可視化の実践")
print("=" * 80)

# ============================================================================
# 2. データの読み込みと基本確認
# ============================================================================

print("\n[1] データの読み込み")
print("-" * 80)

# Google Sheetsからデータ読み込み
url = 'https://docs.google.com/spreadsheets/d/1bVYTLljoSaw8lbJIfsV7n2PcTE3HjJHZwWQ-zqNObpA/export?format=csv'
df = pd.read_csv(url)

# データの確認
print("\n■ データの最初の5行:")
print(df.head())

print(f"\n■ データ形状: {df.shape}")
print(f"  - 学生数: {len(df)}名")
print(f"  - 科目数: {len(df.columns) - 1}科目")

print("\n■ 基本統計量:")
print(df.describe())

# ============================================================================
# 3. データフォーマット変換の実践
# ============================================================================

print("\n" + "=" * 80)
print("[2] データフォーマット変換")
print("=" * 80)

# 3-1. ワイド型からロング型への変換
print("\n■ ワイド型 → ロング型 変換（melt）")
print("-" * 80)

# サンプルデータでデモンストレーション
sample_wide = df.head(3).copy()
print("\n元のワイド型データ:")
print(sample_wide)

# melt()でロング型に変換（TotalScoreを除く科目のみ）
sample_long = sample_wide.melt(
    id_vars=['ID'],
    value_vars=['Math', 'Science', 'English', 'History', 'Art'],
    var_name='Subject',
    value_name='Score'
)
print("\nロング型に変換後:")
print(sample_long)

# 3-2. ロング型からワイド型への変換
print("\n\n■ ロング型 → ワイド型 変換（pivot）")
print("-" * 80)

# pivot()でワイド型に戻す
sample_wide_restored = sample_long.pivot(
    index='ID',
    columns='Subject',
    values='Score'
).reset_index()

print("\nワイド型に戻した結果:")
print(sample_wide_restored)

# ============================================================================
# 4. 可視化手法1: 棒グラフ（個別学生の比較）
# ============================================================================

print("\n" + "=" * 80)
print("[3] 可視化1: 棒グラフ（個別学生の科目別成績）")
print("=" * 80)

# 最初の5人の学生データを抽出
students_sample = df.head(5)[['ID', 'Math', 'Science', 'English', 'History', 'Art']].copy()
students_sample = students_sample.set_index('ID')

# 棒グラフの作成
fig, ax = plt.subplots(figsize=(12, 6))
students_sample.plot(kind='bar', ax=ax, width=0.8)

# グラフの装飾
ax.set_title('Individual Subject Scores Comparison (ID 1-5)', fontsize=16, pad=20)
ax.set_xlabel('Student ID', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(50, 100)
ax.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualization_1_bar_chart.png', dpi=300, bbox_inches='tight')
print("\n✓ 棒グラフを保存しました: visualization_1_bar_chart.png")
plt.close()

# ============================================================================
# 5. 可視化手法2: 相関行列（科目間の関係性）
# ============================================================================

print("\n" + "=" * 80)
print("[4] 可視化2: 相関行列ヒートマップ（科目間の関係性）")
print("=" * 80)

# 数値データのみを抽出（IDとTotalScoreを除く科目のみ）
subjects = df[['Math', 'Science', 'English', 'History', 'Art']]

# 相関係数の計算
correlation_matrix = subjects.corr()

print("\n■ 科目間の相関係数:")
print(correlation_matrix)

# ヒートマップの作成
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,          # 数値を表示
    fmt='.2f',           # 小数点2桁
    cmap='RdBu_r',       # 色マップ（赤-青）
    center=0.5,          # 中心値
    square=True,         # 正方形のセル
    linewidths=1,        # セル間の線
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=ax
)

ax.set_title('Subject Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('visualization_2_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ 相関行列を保存しました: visualization_2_correlation_matrix.png")
plt.close()

# ============================================================================
# 6. 可視化手法3: PCA（主成分分析）
# ============================================================================

print("\n" + "=" * 80)
print("[5] 可視化3: PCA（主成分分析による全体構造の把握）")
print("=" * 80)

# データの準備（科目のみ）
X = df[['Math', 'Science', 'English', 'History', 'Art']]
student_ids = df['ID']

# データの標準化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行（2次元に削減）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 寄与率の確認
print(f"\n■ PCA結果:")
print(f"  第1主成分の寄与率: {pca.explained_variance_ratio_[0]:.3f}")
print(f"  第2主成分の寄与率: {pca.explained_variance_ratio_[1]:.3f}")
print(f"  累積寄与率: {sum(pca.explained_variance_ratio_):.3f}")

# 学生タイプの分類（成績による）
def classify_student(row):
    avg = row.mean()
    if avg >= 90:
        return 'Excellent'
    elif avg >= 85:
        return 'Science_Strong' if row['Science'] > row.mean() else 'History/Art_Strong'
    elif avg >= 75:
        return 'Liberal/Arts_Strong'
    else:
        return 'Below_Average'

student_types = X.apply(classify_student, axis=1)

print(f"\n■ 学生タイプの分布:")
print(student_types.value_counts())

# 散布図の作成
fig, ax = plt.subplots(figsize=(12, 8))

for stype in student_types.unique():
    mask = student_types == stype
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=stype,
        alpha=0.7,
        s=100
    )

ax.set_xlabel(
    f'Principal Component 1 (PC1) - Explained Variance: {pca.explained_variance_ratio_[0]:.2f}',
    fontsize=12
)
ax.set_ylabel(
    f'Principal Component 2 (PC2) - Explained Variance: {pca.explained_variance_ratio_[1]:.2f}',
    fontsize=12
)
ax.set_title('PCA Result (Colored by Student Type)', fontsize=16, pad=20)
ax.legend(title='Student Type')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_3_pca.png', dpi=300, bbox_inches='tight')
print("\n✓ PCA散布図を保存しました: visualization_3_pca.png")
plt.close()

# ============================================================================
# 7. 可視化手法4: 散布図行列（Pairplot）
# ============================================================================

print("\n" + "=" * 80)
print("[6] 可視化4: 散布図行列（全科目ペアの関係性）")
print("=" * 80)

# 学生タイプを追加してデータフレームを作成
df_with_type = df.copy()
df_with_type['Student_Type'] = student_types

# 散布図行列の作成
pairplot = sns.pairplot(
    df_with_type,
    vars=['Math', 'Science', 'English', 'History', 'Art'],
    hue='Student_Type',
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 50},
    diag_kws={'bins': 20, 'alpha': 0.7}
)

pairplot.fig.suptitle('Pairwise Relationships Between Subjects', y=1.01, fontsize=16)
plt.tight_layout()
plt.savefig('visualization_4_pairplot.png', dpi=300, bbox_inches='tight')
print("\n✓ 散布図行列を保存しました: visualization_4_pairplot.png")
plt.close()

# ============================================================================
# 8. 統計的検定: 相関の有意性検定
# ============================================================================

print("\n" + "=" * 80)
print("[7] 統計的検定: 科目間の相関の有意性")
print("=" * 80)

def test_correlation(data, col1, col2, alpha=0.05):
    """2つの科目の相関係数と有意性を検定"""
    corr, p_value = stats.pearsonr(data[col1], data[col2])
    is_significant = "有意" if p_value < alpha else "有意でない"
    return corr, p_value, is_significant

# すべての科目ペアで検定
subjects_list = ['Math', 'Science', 'English', 'History', 'Art']

print("\n■ 科目間の相関と有意性（有意水準α=0.05）:")
print("-" * 80)

results = []
for i, subj1 in enumerate(subjects_list):
    for subj2 in subjects_list[i+1:]:
        corr, p_val, sig = test_correlation(df, subj1, subj2)
        results.append({
            '科目ペア': f'{subj1} vs {subj2}',
            '相関係数': f'{corr:.3f}',
            'p値': f'{p_val:.4f}',
            '有意性': sig
        })
        print(f"{subj1:6s} vs {subj2:6s}: r={corr:6.3f}, p={p_val:.4f} ({sig})")

# 結果をDataFrameとして保存
results_df = pd.DataFrame(results)
results_df.to_csv('correlation_test_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ 検定結果を保存しました: correlation_test_results.csv")

# ============================================================================
# 9. 追加分析: 科目別の記述統計
# ============================================================================

print("\n" + "=" * 80)
print("[8] 追加分析: 科目別の記述統計")
print("=" * 80)

print("\n■ 各科目の詳細統計:")
print("-" * 80)

subject_stats = subjects.describe().T
subject_stats['range'] = subject_stats['max'] - subject_stats['min']
subject_stats['cv'] = subject_stats['std'] / subject_stats['mean']  # 変動係数

print(subject_stats[['mean', 'std', 'min', 'max', 'range', 'cv']])

subject_stats.to_csv('subject_statistics.csv', encoding='utf-8-sig')
print("\n✓ 科目別統計を保存しました: subject_statistics.csv")

# ============================================================================
# 10. まとめ
# ============================================================================

print("\n" + "=" * 80)
print("分析完了！")
print("=" * 80)

print("\n■ 生成されたファイル:")
print("  1. visualization_1_bar_chart.png - 棒グラフ（個別学生比較）")
print("  2. visualization_2_correlation_matrix.png - 相関行列ヒートマップ")
print("  3. visualization_3_pca.png - PCA散布図")
print("  4. visualization_4_pairplot.png - 散布図行列")
print("  5. correlation_test_results.csv - 相関検定結果")
print("  6. subject_statistics.csv - 科目別統計情報")

print("\n■ 主な知見:")
print(f"  - 最も相関が高い科目ペア: {correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().index[1]}")
print(f"  - PC1とPC2の累積寄与率: {sum(pca.explained_variance_ratio_):.1%}")
print(f"  - 学生タイプの分布: {len(student_types.unique())}種類")

print("\n" + "=" * 80)
print("すべての分析が正常に完了しました！")
print("=" * 80)
