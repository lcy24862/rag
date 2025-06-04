# README.md
# 机电系统故障诊断与维修案例教程 RAG系统
知识自动化课设作业，仅供参考  
这是一个基于RAG（检索增强生成）技术的机电系统故障诊断与维修案例教程问答系统。

## 功能特点

- 基于教材内容的智能问答
- 故障诊断信息检索
- 维修步骤指导
- 相似案例推荐

## 安装说明

1. 克隆项目
```bash
git clone https://github.com/lcy24862/rag/
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
```bash
将教材PDF文件放入 `data/textbook/` 目录  
```

5. 创建配置文件config.py
```bash
CONFIG = {
    "data_path": "data/textbook/textbook.pdf",
    "chunk_size": 800,
    "chunk_overlap": 500,
    "embedding_model": "./models/text2vec-base-chinese",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your api key",
    "vector_store": "chroma",
    "llm_model": "deepseek-chat",
    "temperature": 0.7,
    "top_k": 3,
    "vector_store_path": "./chroma_db",
}
```  
5. 运行系统
```bash
python initialize.py
python main.py
```

## 使用说明

1. 启动系统后，输入您的问题
2. 系统会基于教材内容给出相关回答
3. 输入 'quit' 退出系统

## 项目结构

- data/: 存放原始教材数据
- chroma_db/: 向量数据库存储
- src/: 源代码
- config.py: 配置文件
- main.py: 主程序入口

## 注意事项

- 确保已安装所有依赖包
- 确保教材PDF文件已正确放置
- 首次运行可能需要较长时间进行向量化处理
