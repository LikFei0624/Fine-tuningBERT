import chardet

# 打开文件，读取一部分字节来检测编码
with open('/Users/likfei_0624/Desktop/Project I/reviews data.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))  # 读取前10000字节来猜测编码

print(result['encoding'])  # 输出检测到的编码