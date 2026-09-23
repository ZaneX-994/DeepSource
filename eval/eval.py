# -*- coding: utf-8 -*-
"""
基于 Ragas 的 RAG（检索增强生成）评估程序

程序功能：
    1. 读取评估测试集 eval/qa.csv（question、ground_truth）
    2. 逐条调用 RAG 查询流程（query_app.invoke），生成 answer 并从 state 的 reranked_docs 提取 context
    3. 使用 Ragas 评估 5 个指标：
       Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness
    4. 将评估结果写入 eval/result.csv（utf-8 BOM 编码）

运行方式（在项目根目录，需先 uv sync / uv add ragas 安装 ragas）：
    venv/bin/python eval/eval.py
"""

import os
import sys
import warnings

import pandas as pd
from langchain_core.embeddings import Embeddings

# 将项目根目录加入 sys.path，保证可以正常导入 app 包（因为脚本位于 eval 子目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.infra.llm.providers import llm_provider  # noqa: E402
from app.process.query.agent.main_graph import query_app  # noqa: E402
from app.process.query.agent.state import create_query_default_state  # noqa: E402

# ============================================================
# 常量定义
# ============================================================

# 当前评估目录（即 eval.py 所在的 eval 目录）
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
# 评估测试集路径（question、ground_truth）
QA_CSV_PATH = os.path.join(EVAL_DIR, "qa.csv")
# 评估结果输出路径
RESULT_CSV_PATH = os.path.join(EVAL_DIR, "result.csv")

# 结果 CSV 的 9 个列名（与业务要求完全一致）
RESULT_COLUMNS = [
    "question",          # 问题
    "context",           # 检索上下文（来自 reranked_docs）
    "answer",            # RAG 生成的答案
    "ground_truth",      # 参考答案
    "Faithfulness",      # 忠实度
    "Answer Relevancy",  # 答案相关性
    "Context Precision", # 上下文精确率
    "Context Recall",    # 上下文召回率
    "Answer Correctness",# 答案正确性
]


class BgeM3LangchainEmbeddings(Embeddings):
    """
    BGE-M3 向量模型的 LangChain Embeddings 适配器。

    说明：
        Ragas 需要 LangChain 标准的 embedding 接口（embed_documents / embed_query），
        而项目中的 llm_provider.bge_m3_embedding() 返回的是 pymilvus 的
        BGEM3EmbeddingFunction（提供 encode_documents / encode_queries），
        因此这里做一层适配：内部调用项目的 embedding 客户端，对外暴露 LangChain 接口。
    """

    def __init__(self):
        # 懒加载：首次真正编码时才初始化 BGE-M3 模型（模型加载较重，避免重复加载）
        self._bge_m3_ef = None

    def _get_embedding_function(self):
        """获取（懒加载）BGE-M3 模型实例，内部通过项目的 embedding 客户端获得。"""
        if self._bge_m3_ef is None:
            # 通过项目统一的 embedding 客户端获取 BGE-M3 模型
            self._bge_m3_ef = llm_provider.bge_m3_embedding()
        return self._bge_m3_ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本，返回稠密向量列表（LangChain 标准接口）。"""
        ef = self._get_embedding_function()
        # encode_documents 返回 {'dense': 稠密向量列表, 'sparse': 稀疏向量}
        result = ef.encode_documents(texts)
        # dense 中的每个向量是 numpy 数组，逐条转成 list[float]
        return [vec.tolist() for vec in result["dense"]]

    def embed_query(self, text: str) -> list[float]:
        """编码查询文本，返回单个稠密向量（LangChain 标准接口）。"""
        ef = self._get_embedding_function()
        # encode_queries 返回 {'dense': 稠密向量列表, 'sparse': 稀疏向量}
        result = ef.encode_queries([text])
        return result["dense"][0].tolist()


def step_1_load_test_set(csv_path: str) -> pd.DataFrame:
    """
    步骤 1：读取评估测试集。

    读取 eval/qa.csv，并校验必须包含 question、ground_truth 两列。

    :param csv_path: 测试集 CSV 文件路径
    :return: 包含 question、ground_truth 两列的 DataFrame
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 校验必要的列是否存在，缺失则直接抛出异常
    for col in ["question", "ground_truth"]:
        if col not in df.columns:
            raise ValueError(f"评估测试集缺少必要列：{col}")
    return df


def step_2_build_llm():
    """
    步骤 2：获取评估用的大模型客户端。

    直接使用项目中的 llm_provider.chat() 获取 ChatOpenAI 实例；
    ragas 的 evaluate() 会自动将其包装为内部可用的 LLM。
    """
    return llm_provider.chat()


def step_3_build_embeddings() -> Embeddings:
    """
    步骤 3：构建评估用的 embedding 客户端。

    使用 BgeM3LangchainEmbeddings 适配器，其内部通过
    llm_provider.bge_m3_embedding() 获得 BGE-M3 模型。
    """
    return BgeM3LangchainEmbeddings()


def step_4_invoke_rag(question: str, session_id: str) -> dict:
    """
    步骤 4：调用 RAG 查询流程，返回执行完成后的 state。

    使用 query_app.invoke() 同步执行完整查询链路（非流式）。

    :param question: 用户问题
    :param session_id: 会话唯一标识（每个问题独立，避免历史记录串扰）
    :return: 执行完成后的 state（含 answer、reranked_docs 等字段）
    """
    # 封装初始 state：非流式执行
    query_state = create_query_default_state(
        session_id=session_id,
        original_query=question,
        is_stream=False,
    )
    result_state = query_app.invoke(query_state)
    return result_state


def step_5_extract_context(result_state: dict) -> list[str]:
    """
    步骤 5：从 state 的 reranked_docs 字段提取上下文文本列表。

    reranked_docs 中每个文档是 dict，含 text、title、score、type、url 等字段，
    这里只提取 text 作为检索上下文。

    :param result_state: RAG 执行完成后的 state
    :return: 上下文文本列表
    """
    reranked_docs = result_state.get("reranked_docs", []) or []
    contexts = [doc.get("text", "") for doc in reranked_docs]
    return contexts


def step_6_build_samples(test_df: pd.DataFrame):
    """
    步骤 6：逐条执行 RAG，构建 Ragas 样本与原始记录。

    :param test_df: 测试集 DataFrame（question、ground_truth）
    :return: (samples, records)
             samples：Ragas SingleTurnSample 列表
             records：与样本一一对应的原始记录列表（question/context/answer/ground_truth）
    """
    samples = []
    records = []

    for index, row in test_df.iterrows():
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])

        # 每个问题使用独立的 session_id，避免历史记录串扰
        session_id = f"eval_ragas_{index}"

        # 调用 RAG 流程，得到 answer 与 reranked_docs；单条失败不影响整体评估
        try:
            result_state = step_4_invoke_rag(question, session_id)
            answer = result_state.get("answer", "") or ""
            contexts = step_5_extract_context(result_state)
        except Exception as exc:
            # 单条执行失败时记录告警，并置空 answer / context，继续评估其余问题
            print(f"[警告] 第 {index} 条问题执行 RAG 失败：{question}，错误信息：{exc}")
            answer = ""
            contexts = []

        # 组装 Ragas 单轮样本（字段名与 ragas 0.4 保持一致）
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)

        # 记录原始数据，多段 context 用换行符拼接为单个字符串（便于写入 CSV）
        records.append({
            "question": question,
            "context": "\n".join(contexts),
            "answer": answer,
            "ground_truth": ground_truth,
        })

    return samples, records


def step_7_build_metrics() -> list:
    """
    步骤 7：构建 5 个 Ragas 评估指标。

    指标：
        Faithfulness      —— 忠实度（答案是否忠实于检索上下文）
        AnswerRelevancy   —— 答案相关性（答案是否切题）
        ContextPrecision  —— 上下文精确率（检索到的上下文是否相关且排序合理）
        ContextRecall     —— 上下文召回率（参考答案中的信息是否被检索上下文覆盖）
        AnswerCorrectness —— 答案正确性（答案与参考答案的一致性）
    """
    return [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
    ]


def step_8_run_evaluation(samples, metrics, llm, embeddings):
    """
    步骤 8：运行 Ragas 评估，返回评估结果对象。

    llm / embeddings 会由 evaluate() 自动包装并注入到需要它们的指标中。

    :param samples: Ragas 样本列表
    :param metrics: Ragas 指标列表
    :param llm: 大模型客户端（llm_provider.chat() 返回值）
    :param embeddings: embedding 客户端（BgeM3LangchainEmbeddings 实例）
    :return: EvaluationResult 评估结果对象
    """
    # 忽略 ragas evaluate() 的弃用告警，避免输出干扰
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
    return result


def step_9_build_result_df(records: list, scores: list) -> pd.DataFrame:
    """
    步骤 9：组装最终的评估结果 DataFrame（9 列）。

    :param records: 原始记录列表（question/context/answer/ground_truth）
    :param scores: ragas 返回的分数列表（每个元素是 {指标名: 分数} 字典）
    :return: 9 列的评估结果 DataFrame
    """
    rows = []
    for rec, score in zip(records, scores):
        # 将 ragas 内部指标名映射为业务要求的列名
        row = {
            "question": rec["question"],
            "context": rec["context"],
            "answer": rec["answer"],
            "ground_truth": rec["ground_truth"],
            "Faithfulness": score.get("faithfulness"),
            "Answer Relevancy": score.get("answer_relevancy"),
            "Context Precision": score.get("context_precision"),
            "Context Recall": score.get("context_recall"),
            "Answer Correctness": score.get("answer_correctness"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def step_10_save_result(result_df: pd.DataFrame, csv_path: str) -> None:
    """
    步骤 10：将评估结果保存为 CSV 文件（utf-8 BOM 编码）。

    :param result_df: 评估结果 DataFrame
    :param csv_path: 输出 CSV 文件路径
    """
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"评估结果已保存至：{csv_path}")


def main():
    """评估程序主入口：串联全部步骤执行。"""
    print("=" * 60)
    print("开始 Ragas 评估")

    # 步骤 1：读取评估测试集
    test_df = step_1_load_test_set(QA_CSV_PATH)
    print(f"步骤 1 完成：读取测试集 {len(test_df)} 条")

    # 步骤 2 / 3：构建 LLM 与 embedding 客户端
    llm = step_2_build_llm()
    embeddings = step_3_build_embeddings()
    print("步骤 2 / 3 完成：构建 LLM 与 embedding 客户端")

    # 步骤 4 / 5 / 6：逐条执行 RAG 并构建样本
    samples, records = step_6_build_samples(test_df)
    print(f"步骤 4 / 5 / 6 完成：执行 RAG 并构建 {len(samples)} 个样本")

    # 步骤 7：构建 5 个评估指标
    metrics = step_7_build_metrics()
    print("步骤 7 完成：构建 5 个 Ragas 指标")

    # 步骤 8：运行 Ragas 评估
    result = step_8_run_evaluation(samples, metrics, llm, embeddings)
    print("步骤 8 完成：Ragas 评估执行完毕")

    # 步骤 9：组装结果 DataFrame
    result_df = step_9_build_result_df(records, result.scores)
    print("步骤 9 完成：组装结果 DataFrame")

    # 步骤 10：保存结果
    step_10_save_result(result_df, RESULT_CSV_PATH)

    print("评估结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
