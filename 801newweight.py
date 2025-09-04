import pandas as pd
import hashlib
import random
import os
from collections import defaultdict, Counter
import torch
import time
import sys
from typing import List, Dict, Tuple, Any

try:
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
except ImportError:
    print("错误：请确保安装了transformers库：pip install transformers")
    print("同时需要安装protobuf和sentencepiece：pip install protobuf sentencepiece")
    exit(1)


# 清除代理设置
def clear_proxies():
    """清除所有可能的代理设置，确保模型正常下载"""
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    # 对于Windows系统，额外检查系统代理设置
    if sys.platform.startswith('win'):
        try:
            import winreg
            reg_path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, 'ProxyEnable', 0, winreg.REG_DWORD, 0)
        except Exception as e:
            print(f"警告: 无法修改Windows注册表代理设置: {e}")


class EnhancedByT5Generator:
    """增强版ByT5生成器，优化文本变体生成质量"""

    def __init__(self, model_size=r"D:\实习\COMP1\ByT5", device="cuda" if torch.cuda.is_available() else "cpu"):
        """初始化模型，自动选择设备"""
        print(f"加载ByT5模型 ({model_size}) 到 {device}...")
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_size)
        self.model = T5ForConditionalGeneration.from_pretrained(model_size).to(device)
        self.device = device
        print(f"ByT5模型加载完成，使用设备: {device}")

        # 根据设备性能调整生成参数
        if torch.cuda.is_available():
            print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
            self.inference_params = {
                "num_beams": 4,
                "num_return_sequences": 4,
                "max_length": 50,
                "temperature": 0.7,
                "top_p": 0.92,
                "repetition_penalty": 1.2  # 减少重复生成
            }
        else:
            print("使用CPU进行推理")
            self.inference_params = {
                "num_beams": 2,
                "num_return_sequences": 2,
                "max_length": 40,
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }

    def generate_meaningful_variants(self, text: str, q: int = 2, min_similarity: float = 0.6) -> List[str]:
        """生成与原始文本语义相似的变体，过滤无意义变体"""
        if not isinstance(text, str) or len(text) < q:
            return [text] if isinstance(text, str) else []

        # 生成变体
        input_text = f"generate semantically similar variations of: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                early_stopping=True, **self.inference_params
            )
        except Exception as e:
            print(f"生成变体时出错: {e}，使用原始文本")
            return [text]

        # 解码并过滤变体
        variants = []
        for output in outputs:
            try:
                variant = self.tokenizer.decode(output, skip_special_tokens=True)
                # 过滤过短或过长的变体
                if 0.5 * len(text) <= len(variant) <= 1.5 * len(text):
                    # 计算与原始文本的简单相似度
                    if self.calculate_similarity(text, variant) >= min_similarity:
                        variants.append(variant)
            except Exception as e:
                print(f"解码失败: {e}")

        # 确保至少返回原始文本
        if not variants:
            return [text]

        # 去重并保留独特变体
        unique_variants = list(set(variants))
        return unique_variants[:self.inference_params["num_return_sequences"]]  # 限制数量

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """计算两个文本的简单相似度，用于过滤无意义变体"""
        if not text1 or not text2:
            return 0.0

        # 使用字符级n-gram计算相似度
        n = 2
        set1 = set([text1[i:i + n] for i in range(len(text1) - n + 1)])
        set2 = set([text2[i:i + n] for i in range(len(text2) - n + 1)])

        if not set1 or not set2:
            return 0.0

        return len(set1 & set2) / len(set1 | set2)


def load_test_data(file_path: str, source_name: str) -> pd.DataFrame:
    """加载测试数据并进行基本清洗"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        df = pd.read_csv(file_path)
        print(f"成功加载 {source_name}: {len(df)} 条记录")

        # 基本数据清洗
        df = df.drop_duplicates()  # 去重
        df = df.dropna(how='all')  # 删除全为空的行

        return df
    except Exception as e:
        print(f"错误：读取 {file_path} 时发生异常 - {str(e)}")
        raise


def extract_text_features(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """从数据框中提取文本特征，合并多个字段"""
    text_features = []

    for _, row in df.iterrows():
        # 合并多个字段的文本
        text_parts = []
        for col in columns:
            if pd.notna(row[col]) and isinstance(row[col], str) and len(row[col].strip()) > 0:
                text_parts.append(row[col].strip())

        if text_parts:
            # 合并为一个字符串，保留字段间区分
            combined_text = " | ".join(text_parts)
            text_features.append(combined_text)

    print(f"从 {len(df)} 条记录中提取了 {len(text_features)} 个有效文本特征")
    return text_features


def calculate_character_weights(texts: List[str], generator: EnhancedByT5Generator,
                                q: int = 2, min_count: int = 3) -> Dict[str, float]:
    """
    计算字符替换权重，优化版本：
    1. 过滤低频替换
    2. 考虑上下文信息
    3. 增加置信度计算
    """
    # 存储替换计数和原始字符总出现次数
    replacement_counts = defaultdict(Counter)  # replacement_counts[original][replacement] = count
    original_char_counts = Counter()
    context_counts = defaultdict(lambda: defaultdict(Counter))  # 考虑上下文的计数

    total_texts = len(texts)
    start_time = time.time()

    for i, text in enumerate(texts):
        # 进度显示
        if (i + 1) % max(1, total_texts // 10) == 0:
            progress = (i + 1) / total_texts * 100
            elapsed = time.time() - start_time
            eta = elapsed * (total_texts / (i + 1) - 1)
            print(f"处理文本 {i + 1}/{total_texts} ({progress:.1f}%)，耗时: {elapsed:.2f}s，预计剩余: {eta:.2f}s")

        # 生成变体
        variants = generator.generate_meaningful_variants(text, q=q)
        if not variants:
            continue

        # 提取原始文本的字符和q-gram
        original_chars = list(text)
        original_qgrams = [text[j:j + q] for j in range(len(text) - q + 1)]

        # 处理每个变体
        for variant in variants:
            if variant == text:  # 跳过与原始文本相同的变体
                continue

            variant_chars = list(variant)
            min_length = min(len(original_chars), len(variant_chars))

            # 比较字符级差异并记录
            for j in range(min_length):
                orig_char = original_chars[j]
                repl_char = variant_chars[j]

                if orig_char != repl_char:
                    # 记录基本替换计数
                    replacement_counts[orig_char][repl_char] += 1
                    original_char_counts[orig_char] += 1

                    # 记录上下文信息（前一个和后一个字符）
                    prev_char = original_chars[j - 1] if j > 0 else "<START>"
                    next_char = original_chars[j + 1] if j < len(original_chars) - 1 else "<END>"
                    context = f"{prev_char}_{next_char}"
                    context_counts[orig_char][context][repl_char] += 1

        # 处理q-gram级别替换
        for variant in variants:
            if variant == text:
                continue

            variant_qgrams = [variant[j:j + q] for j in range(len(variant) - q + 1)]
            min_qgram_length = min(len(original_qgrams), len(variant_qgrams))

            for j in range(min_qgram_length):
                orig_qgram = original_qgrams[j]
                repl_qgram = variant_qgrams[j]

                if orig_qgram != repl_qgram:
                    # 比较q-gram中的每个字符
                    for k in range(min(len(orig_qgram), len(repl_qgram))):
                        o_char = orig_qgram[k]
                        r_char = repl_qgram[k]
                        if o_char != r_char:
                            replacement_counts[o_char][r_char] += 1
                            original_char_counts[o_char] += 1

    # 计算最终权重，过滤低频替换
    weights = {}
    for orig_char, replacements in replacement_counts.items():
        total = original_char_counts[orig_char]
        if total < min_count:  # 过滤出现次数太少的字符
            continue

        for repl_char, count in replacements.items():
            if count < min_count:  # 过滤低频替换
                continue

            # 基础权重：替换频率
            base_weight = count / total

            # 考虑上下文的置信度调整
            context_confidence = 1.0
            contexts = context_counts.get(orig_char, {})
            if contexts:
                total_context_count = sum(sum(ctx.values()) for ctx in contexts.values())
                if total_context_count > 0:
                    # 计算上下文一致性得分
                    context_scores = []
                    for ctx, char_counts in contexts.items():
                        ctx_total = sum(char_counts.values())
                        if ctx_total > 0:
                            ctx_score = char_counts.get(repl_char, 0) / ctx_total
                            context_scores.append(ctx_score)

                    if context_scores:
                        context_confidence = sum(context_scores) / len(context_scores)

            # 最终权重：基础权重 × 上下文置信度
            final_weight = base_weight * context_confidence
            weights[f"{orig_char}-{repl_char}"] = min(1.0, final_weight)  # 确保不超过1.0

    print(f"计算完成，共得到 {len(weights)} 个有效字符替换权重")
    return weights


def save_weights(weights: Dict[str, float], output_file: str) -> None:
    """保存权重到CSV文件"""
    # 转换为DataFrame
    data = []
    for key, weight in weights.items():
        orig, repl = key.split('-', 1)  # 分割原始字符和替换字符
        data.append({
            'original': orig,
            'replacement': repl,
            'weight': weight,
            'confidence': min(1.0, weight * 2)  # 简单计算置信度
        })

    df = pd.DataFrame(data)

    # 按权重降序排序
    df = df.sort_values('weight', ascending=False)

    # 保存文件
    df.to_csv(output_file, index=False)
    print(f"权重文件已保存至: {output_file}，包含 {len(df)} 个替换规则")


def main():
    # 设置随机种子，确保结果可复现
    random.seed(9318)
    torch.manual_seed(9318)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9318)

    start_time = time.time()

    # 清除代理设置
    clear_proxies()

    # 配置参数
    q = 2  # q-gram长度
    min_replacement_count = 3  # 最小替换计数，过滤噪声
    max_training_samples = 1000  # 最大训练样本数，控制计算量
    text_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                    'right_title', 'right_authors', 'right_venue', 'right_year']  # 需要提取的文本列

    # 文件路径设置
    data_dir = r"D:\实习\COMP1\icip2"
    test_files = [
        os.path.join(data_dir, "dirty_dblp_acm", "test.csv"),
        os.path.join(data_dir, "dirty_dblp_scholar", "test.csv")
    ]

    # 确保结果目录存在
    result_dir = os.path.join(data_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # 输出权重文件路径
    output_weight_file = os.path.join(result_dir, 'optimized_weights_from_test.csv')

    try:
        # 加载测试数据
        print("加载测试数据集...")
        all_text_features = []
        for i, file_path in enumerate(test_files):
            source_name = f"test_data_{i + 1}"
            df = load_test_data(file_path, source_name)
            # 提取文本特征
            text_features = extract_text_features(df, text_columns)
            all_text_features.extend(text_features)

        # 限制样本数量，平衡精度和计算时间
        if len(all_text_features) > max_training_samples:
            print(f"从 {len(all_text_features)} 个文本特征中随机选择 {max_training_samples} 个用于权重计算")
            all_text_features = random.sample(all_text_features, max_training_samples)

        # 初始化增强版ByT5生成器
        generator = EnhancedByT5Generator()

        # 计算字符替换权重
        print("\n开始计算字符替换权重...")
        weights = calculate_character_weights(
            texts=all_text_features,
            generator=generator,
            q=q,
            min_count=min_replacement_count
        )

        # 保存权重文件
        save_weights(weights, output_weight_file)

    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 输出总运行时间
    total_time = time.time() - start_time
    print(f"\n权重计算完成，总耗时: {total_time:.2f}s")
    print(f"生成的权重文件: {output_weight_file}")


if __name__ == "__main__":
    main()
