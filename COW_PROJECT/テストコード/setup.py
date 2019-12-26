#-*- encoding:utf-8 -*-

import sys, os
from cx_Freeze import setup, Executable

file_path = input("アプリ化したいpy：")
base = None
packages = []

includes = [
    "sys",
    "os",
    "requests",
    "numpy",
    "pandas",
    "community",
    "networkx",
    "codecs"
]

excludes = []

exe = Executable(
    script = file_path,
    base = base
)
 
# セットアップ
setup(name = 'main',
      options = {
          "build_exe": {
              "packages": packages, 
              "includes": includes, 
              "excludes": excludes,
          }
      },
      version = '0.1',
      description = 'converter',
      executables = [exe])