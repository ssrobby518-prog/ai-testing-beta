#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def modify_cumulative_excel():
    file_path = 'output/data/cumulative.xlsx'
    try:
        df = pd.read_excel(file_path)
        
        # 新增欄位，如果不存在
        if '是否為AI影片' not in df.columns:
            df['是否為AI影片'] = ''
        
        # 根據用戶指定標註
        for i, row in df.iterrows():
            file_name = row.get('File Path', '')  # 假設欄位名為 'File Path'
            if '0903-0006.mp4' in file_name:
                df.at[i, '是否為AI影片'] = 'Y'
            elif '0911-005.mp4' in file_name:
                df.at[i, '是否為AI影片'] = 'N'
            # 其他檔案留空或根據需要擴展
        
        # 寫回檔案
        df.to_excel(file_path, index=False)
        print("Excel 修改完成：新增/更新 '是否為AI影片' 欄位。")
    except Exception as e:
        print(f"錯誤：{e}")

if __name__ == "__main__":
    modify_cumulative_excel()