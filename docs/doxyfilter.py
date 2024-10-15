import sys
import re

def preprocess(line):
    line = re.sub(r'PYBIND11_MODULE\(([^,]+),\s*([^,]+)\)', r'void \1()', line)
    return line

if __name__ == "__main__":
    for line in sys.stdin:
        print(preprocess(line), end='')