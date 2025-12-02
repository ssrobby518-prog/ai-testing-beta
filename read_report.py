#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Error reading Excel: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_report.py <excel_path>")
    else:
        read_excel(sys.argv[1])