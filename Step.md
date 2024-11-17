
## 得到html的文本

[Beautiful Soup 4.12.0 文档 — Beautiful Soup 4.12.0 documentation (crummy.com)](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/)

[BeautifulSoup 三个方法：getText()、text()和get_text()|极客教程 (geek-docs.com)](https://geek-docs.com/beautifulsoup/beautifulsoup-questions/245_beautifulsoup_gettext_vs_text_vs_get_text.html)

`html2text.py`

## 把Ofiicial Account Name、标题、评论编码

把Ofiicial Account Name、标题、评论使用[句子编码器](https://huggingface.co/sentence-transformers/use-cmlm-multilingual)编码
让后存到numpy文件或者h5py文件中

> 文本编码的模型大小不能太大，最好不要超过1GB。

[jinaai/jina-embeddings-v2-base-zh · Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh)

[shibing624/text2vec-base-中文 ·拥抱脸 (huggingface.co)](https://huggingface.co/shibing624/text2vec-base-chinese)

[uer/sbert-base-chinese-nli · Hugging Face](https://huggingface.co/uer/sbert-base-chinese-nli)

> 不使用多语言的

> 使用bert的暂时不考虑

[google-bert/bert-base-chinese · Hugging Face](https://huggingface.co/google-bert/bert-base-chinese)

[ckiplab/bert-base-chinese-ner ·拥抱脸 (huggingface.co)](https://huggingface.co/ckiplab/bert-base-chinese-ner)

> 这里先不考虑用html里的内容，先把流程跑通
## dataset读取

从numpy文件或者h5py文件读取

数据集很小，所以都读到内存

## 找一找有没有针对长文本的编码器

[yongzhuo/Pytorch-NLU: Pytorch-NLU，一个中文文本分类、序列标注工具包，支持中文长文本、短文本的多类、多标签分类任务，支持中文命名实体识别、词性标注、分词、抽取式文本摘要等序列标注任务。 Ptorch NLU, a Chinese text classification and sequence annotation toolkit, supports multi class and multi label classification tasks of Chinese long text and short text, and supports sequence annotation tasks such as Chinese named entity recognition, part of spee (github.com)](https://github.com/yongzhuo/Pytorch-NLU)

[qingyujean/document-level-classification: 超长文本分类（大于1000字）；文档级/篇章级文本分类；主要是解决长距离依赖问题 (github.com)](https://github.com/qingyujean/document-level-classification/tree/main)


https://blog.csdn.net/jclian91/article/details/129658324

[NLP实战：Pytorch实现7大经典深度学习中文文本分类-TextCNN+TextRNN+FastText+TextRCNN+TextRNN_Attention+DPCNN+Transformer_基于pytorch深度学习的nlp代码-CSDN博客](https://blog.csdn.net/qq_31136513/article/details/131589556)

- DPCNN 看一看
- XLNet + 层次attention + fc 不太用看
- DeepSeek 可以看一看

## 根据文章的新闻来源可以考虑使用MoE架构？

使用 `朝阳实拍、时事内幕爆料` 等经过CLIP编码，得到 编码后 ，用这一段 作为MoE(DeepSeek)的Gate的输入。
