# -*- coding: utf-8 -*- 
# @File sm4.py
# @Time 2021/3/18 12:19
# @Author wcy
# @Software: PyCharm
# @Site
import time
from pysm4 import encrypt_ecb, decrypt_ecb



if __name__ == '__main__':
    app_id = "f7ba1b40e6f2436a92b75843f2b175eb"
    secret = "b5a11bd5610d4eac"
    timestamp = int(round(time.time() * 1000))

    # 明文
    plain_text = f"{app_id}:{timestamp}"
    # 密钥
    key = secret  # 密钥长度小于等于16字节
    # 加密
    cipher_text = encrypt_ecb(plain_text, key)
    print(cipher_text)
    # 解密
    print(plain_text == decrypt_ecb(cipher_text, key))