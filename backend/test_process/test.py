import pandas as pd

# 設定 CSV 檔案的路徑
csv_file_path = '/home/user/PromptSDVis/data/diffusiondb_data/random_100k/diffusiondb_random_100k/imgs_rand100k_with_keywords.csv'

# 使用 Pandas 讀取 CSV 檔案
df = pd.read_csv(csv_file_path)

# 獲取行數（不包括標頭）
row_count = len(df)
print(f'CSV 檔案有 {row_count} 行資料')

# 設置 pandas 顯示選項，確保資料不會被截斷


# 只顯示特定欄位 'prompt' 和 'keywords'
print(df[['prompt', 'keywords']].head(10))
