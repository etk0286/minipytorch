# main.py
import sys
import numpy as np

def main():
    print("=== Graph Engine Environment Setup Successful ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version: {np.__version__}")
    print("ready to start writing Node classes")

if __name__ == "__main__":
    main()