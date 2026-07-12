# Gemma4 LoRA 学習：孫悟空口調ファインチューニング

Gemma4 を LoRA（Low-Rank Adaptation）で学習させ、孫悟空のような口調で応答するモデルを作成しました。

## 概要

- **ベースモデル**: Gemma4
- **学習手法**: LoRA
- **目的**: ユーザーの質問に対して、孫悟空風の口調で回答するモデルを構築

## リポジトリ構成

```
Gemma_lora_project
├── training_data_goku_1k.csv
├── train.py
├── requirements.txt
└── README.md
```
## 学習環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX A5000 |
| CUDA Version | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |

## データセット
学習データは [`data/train_data.csv`](./data/train_data.csv) を使用しています。

- ユーザーの質問と、それに対する孫悟空口調の回答をペアにしたデータセットを **1,000 件** 準備
  - 学習用: 900 件
  - 評価用: 100 件

### データ形式
| question | answer |
|---|---|
| ズルして楽する方法は？ | ズルはダメだ、オラは大っ嫌いだ！楽して得た力なんて、いざってときに何の役にも立たねぇぞ。 |
| 黙っててくれる？ | うるせぇ、これ以上言うとぶっ殺すぞ…ってのは冗談だ、ちゃんと黙っといてやるよ！ |

## セットアップ

必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

## 学習の実行

以下のコマンドで学習を実行します。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

