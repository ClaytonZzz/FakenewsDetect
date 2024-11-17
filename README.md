
## 官网

[2024年第六届全国高校计算机能力挑战赛官网 - 人工智能挑战赛 (ncccu.org.cn)](https://www.ncccu.org.cn/index/Paper/case1.html)

## 数据集

> [Weak Supervision for Fake News Detection via Reinforcement Learning | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/5389)

> [yaqingwang/WeFEND-AAAI20: Dataset for paper "Weak Supervision for Fake News Detection via Reinforcement Learning" published in AAAI'2020. (github.com)](https://github.com/yaqingwang/WeFEND-AAAI20/tree/master)


## 引用数据集的论文

[Wang： 对假新闻侦测监管不力... - Google 學術搜尋](https://scholar.google.com.hk/scholar?start=0&hl=zh-TW&as_sdt=2005&sciodt=0,5&cites=3571851614042495630&scipsc=)

## 可借鉴的代码或者思路

### 代码

#### 同一个数据集

1. 数据处理 + LSTM（这个LSTM可以用Transformer结构替代）
[TephrocactusHC/PYTHON-HOMEWORK: PYTHON课的期末作业，写的超级烂，当时啥也不懂，随便搞了搞就交上去了。 (github.com)](https://github.com/TephrocactusHC/PYTHON-HOMEWORK/tree/main)
#### 不同数据集

1. BERT编码 + Scaled Dot Product Attention+ MoE的一部分 

[CIKM 2021 | MDFEND：多领域虚假新闻检测（已开源） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/443690475)

## 问题

1. 都没有使用链接里的内容，靠标题和评论进行判断假新闻还是真新闻(这样比较容易做，也先不使用html里的文本)
2. 

## code

- `html2text.py` 提取html文件里的文本
- `text2emb.py` 将train.csv里的文本进行编码，一句话编码维度为`768`
- `dataset.py` 数据集
- `model_base.py transformer_layer.py` 模型
- `config_fn.py` 超参数
- `utils.py` 设置随机种子、EMA训练等一些工具

