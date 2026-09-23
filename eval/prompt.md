### 用于生成ragas评估测试集
【核心目标】
    1 生成一份ragas评估测试集
【要求】
    1  文件格式 csv ，字符集 utf-8 BOM 
    2  文件名：qa.csv 保存在 eval目录下 
    3  列头： question , ground_truth
    4  基于  eval/hak180产品安全手册_new.md生成10 对问题和答案
    5  使用“HAK 180 烫金机：” 作为问题前缀



###  生成评估测试程序 
【核心目标】
    1  生成一个基于ragas评估测试程序 
【环境信息】
    1  评估测试集在 eval/qa.csv 文件中
    2  rag流程入口 在 app/process/query/agent/main_graph.py 中的 query_graph.invoke()
    3  上下文获取从 state 中 reranked_docs字段提取
    4  llm的客户端使用 app/infra/llm/providers.py中 的llm_provider.chat()获得，embedding 的客户端使用 app/infra/llm/providers.py中的llm_provider.bge_m3_embedding() 获得
【业务要求】
    1  基于ragas 评估5个指标: Faithfulness 、Answer Relevancy、Context Precision、Context Recall、Answer Correctness 
    2  程序运行最终生成一个csv的评估结果 名称为 result.csv 保存在 eval目录中 编码集utf-8 bom
    3   result.csv 包含9列：question、context、answer、 ground_truth、  Faithfulness 、Answer Relevancy、Context Precision、Context Recall、Answer Correctness 
【程序要求】
    1  程序名称为 eval.py 保存在eval目录中
    2  程序中每个函数要求有中文注释 
    3  核心步骤函数的命名，要用 step_1_xx、step_2_xx... 命名。


### 优化知识图谱
帮我设计一个知识图谱的实践场景
【背景】
    我希望能够设计一个培训案例，能够说明知识图谱可以解决跨切片语义断裂的问题。
【要求】
     1 帮我基于 eval\hak180产品安全手册_new.md 或 eval\万用表RS-12的使用_new.md 生产几个跨切片语义的用户问题，这种问题在没有知识图谱功能情况下经过评估分数都不会高。
     2 优化当前的知识图谱功能，使得这些问题在有图谱和没图谱下，差异明显，能够突出图谱的作用。
【输出】
     基于以上背景要求，给我写一个方案，包括要生成的问题、当前图谱如何改造。保存在eval中。