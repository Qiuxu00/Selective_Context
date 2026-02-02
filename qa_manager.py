# Use OpenAI's GPT-3.5-turbo to generate questions and answer from a given document
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, asdict
from context_manager import ArxivContext
import sys
import json
import logging
import evaluate
import os
import openai
import time
import pickle
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, pipeline, LlamaForCausalLM, \
                    LlamaTokenizerFast, GenerationConfig, T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
from tqdm import tqdm

@dataclass
class ContextAndAnswer:
    """
        This class stores contexts and its list of masked_contexts.
        It also take care of questions and reference answer based on original context.
        As well as answers based on masked_context.

        should add a function in this class to evaluate/compare answers against reference answer.
    """
    reference_context: str
    contexts_dict: Dict[str, List[ArxivContext]]
    mask_ratio: float
    reduced_ratio: Dict[str, float] = None
    task_name: str = None
    questions: Union[List[str], List[List[str]], Dict[str, List[str]], Dict[str, List[List[str]]]] = None
    answer_of_contexts: Dict[str, List[str]] = None
    dataset_type : str = None
    metrics: Dict[str, float] = None

    def __post_init__(self):
        reference_contexts = self.contexts_dict[self.reference_context]
        self.reduced_ratio = {}
        for context_type in self.contexts_dict:
            if context_type == self.reference_context:
                continue
            self.reduced_ratio[context_type] = []
            for ref, cont in zip(reference_contexts, self.contexts_dict[context_type]):
                sub_len = len(ref.context) - len(cont.context)
                if sub_len < 0:
                    sub_len = 0
                self.reduced_ratio[context_type].append(sub_len / len(ref.context))
    
    def __repr__(self):
        contexts = '\n'.join(self.contexts_dict.keys())
        return f"ContextAndAnswer:\n{contexts}"
            
class TaskManager:

    def __init__(self, task_name, model_type, save_path, only_eval = False, metrics = ['bleu', 'meteor', 'rouge', ]):
        self.task_name = task_name
        self.model_type = model_type
        self.save_path = save_path

        if not only_eval:
            self._prepare_model()
        # self._prepare_evaluation(metrics)
    
    def _prepare_model(self):
        self.model_instruct_tuned = False # 默认初始化为 False
        print(f'-- Start preparing model {self.model_type}.')
        # prepare model and generate function
        # should support GPT-3.5-turbo, llama-7B,13B,30B, and Flan family?
        print(f'-- Start preparing model {self.model_type}.')
        if self.model_type == "gpt-3.5-turbo":
            self.model_instruct_tuned = True
            self._generate_answer = self._gpt_3_5_turbo_generate
        elif 'llama' in self.model_type:
            self.model_instruct_tuned = False
            size = self.model_type.split('-')[-1]
            assert size in ['7b', '13b', '30b']
            bs = {
                '7b': 24,
                '13b': 12,
                '30b': 6,
            }
            self.batch_size = bs[size]
            if size == '30b':
                max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
                n_gpus = torch.cuda.device_count()
                max_memory = {i: max_memory for i in range(n_gpus)}
                self.model = LlamaForCausalLM.from_pretrained(f"huggyllama/llama-{size}", load_in_8bit=True, device_map='auto', max_memory=max_memory, cache_dir="/mnt/fast/nobackup/scratch4weeks/yl02706/HF_Cache")
            else:
                self.model = LlamaForCausalLM.from_pretrained(f"huggyllama/llama-{size}", torch_dtype=torch.float16, device_map='auto')
            self.tokenizer = LlamaTokenizerFast.from_pretrained(f"huggyllama/llama-{size}")
            self.model.eval()

            self.generation_config = GenerationConfig(
                temperature=1.0,
                top_k=50,
                # top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # num_beams=4,
            )

            self._generate_answer = self._lm_generate
        elif self.model_type == 'alpaca-lora-7b':
            self.model_instruct_tuned = True
            base_model = 'huggyllama/llama-7b'
            LORA_WEIGHTS = "tloen/alpaca-lora-7b"
            tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
            model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map='auto')
            model = PeftModel.from_pretrained(model,  torch_dtype=torch.float16)
            model.eval()
            self.batch_size = 24

            self.model = model
            self.tokenizer = tokenizer
            self.generation_config = GenerationConfig(
                temperature=1.0,
                top_k=50,
                # top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # num_beams=4,
            )
            self._generate_answer = self._lm_generate
        elif 'flan' in self.model_type:
            self.model_instruct_tuned = True
            tokenizer = T5Tokenizer.from_pretrained(f"google/{self.model_type}")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.float16, device_map="auto")
            model.eval()
            bs = {
                'flan-t5-xxl': 12,
                'flan-t5-base': 24,
                'flan-t5-large': 24,
                'flan-t5-xl': 24,
            }
            self.batch_size = bs[self.model_type]

            self.model = model
            self.tokenizer = tokenizer
            self.generation_config = GenerationConfig(
                temperature=1.0,
                top_k=50,
                # top_p=0.9,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # num_beams=4,
            )
            self._generate_answer = self._lm_generate
        elif 'gpt2' in self.model_type:
            # === 新增/优化 GPT-2 适配逻辑 ===
            from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
            self.model_instruct_tuned = False 
            self.batch_size = 24
            
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.padding_side = 'left'
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", device_map='auto')
            self.model.eval()

            # 设置 pad_token（GPT-2 默认没有，必须手动指定，否则批处理会报错）
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 1. 使用模型自带配置创建 GenerationConfig，确保兼容性
            g_config = GenerationConfig.from_model_config(self.model.config)
            
            # 2. 更新实验所需的生成参数
            g_config.update(
                temperature=1.0,
                top_k=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=450 # 显式定义，防止 pipeline 默认值过短
            )
            
            # 3. 关键：将配置直接绑定到模型对象上
            # 这样我们在 _lm_answer_batch 里调用 pipeline 时，就不需要再传参数了
            self.model.generation_config = g_config
            self.generation_config = g_config
            
            self._generate_answer = self._lm_generate
            # ========================
        elif 'vicuna' in self.model_type:
            self.model_instruct_tuned = True
            size = self.model_type.split('-')[-1]
            assert size in ['7B', '13B']
            self.batch_size = 12 if size == '13B' else 24
            self.model = LlamaForCausalLM.from_pretrained(f"TheBloke/vicuna-{size}-1.1-HF", torch_dtype=torch.float16, device_map='auto')
            self.tokenizer = LlamaTokenizerFast.from_pretrained(f"huggyllama/llama-{size}".lower())
            self.model.eval()

            self.generation_config = GenerationConfig(
                temperature=1.0,
                top_k=50,
                # top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # num_beams=4,
            )

            self._generate_answer = self._lm_generate
    
    def _lm_generate(self, prompt, num_retry=5):
        # === 修复点 1：输入截断 (核心防爆锁) ===
        # GPT-2 上限 1024。我们预留 500 给生成，输入强制限制在 520 以内。
        # truncation=True 是防止爆仓的关键
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=520)
        input_ids = inputs.input_ids.to(self.model.device)

        with torch.no_grad():
            # === 修复点 2：更安全的配置读取 ===
            # 防止 self.generation_config 未定义
            g_config = getattr(self, 'generation_config', None)
            
            outputs = self.model.generate(
                input_ids, 
                generation_config=g_config, 
                return_dict_in_generate=True, 
                max_new_tokens=500,
                # === 修复点 3：显式指定 pad_token，消除警告 ===
                pad_token_id=self.tokenizer.eos_token_id 
            )
        
        s = outputs.sequences[0]
        prompt_len = input_ids.shape[1]
        # === 修复点 4：解码时跳过特殊符号 (可选优化) ===
        output = self.tokenizer.decode(s[prompt_len:], skip_special_tokens=True)
        return output
    
    def _lm_answer_batch(self, prompts):
        # 1. 确保生成器（Pipeline）已初始化
        if not hasattr(self, 'generator'):
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
            
            g_config = getattr(self, 'generation_config', None)
            if g_config is not None:
                self.model.generation_config = g_config
            
            self.generator = pipeline(
                'text-generation', 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.model.device
            )

        # 2. 【核心保护】在调用显卡前，强制截断所有 Prompt
        # GPT-2 限制 1024。公式：$570 (输入) + 450 (输出) = 1020 \le 1024$
        safe_prompts = []
        for p in prompts:
            # 截断并解码回字符串，确保输入在安全长度内
            # 这里的 truncation=True 是第一道防线
            encoding = self.tokenizer(p, truncation=True, max_length=570)
            safe_prompts.append(self.tokenizer.decode(encoding['input_ids'], skip_special_tokens=True))

        print(f'Batched generation started. num_prompts: {len(safe_prompts)}, batch_size: {self.batch_size}')
        
        # 3. 【唯一调用】使用安全提示词进行批处理生成
        outputs = self.generator(
            safe_prompts, 
            max_new_tokens=450, 
            batch_size=self.batch_size, 
            return_full_text=False,
            # 这里的 truncation 是针对 pipeline 内部的第二道防线
            truncation=True 
        )
        
        return [output[0]['generated_text'] for output in outputs]
    
    def _gpt_3_5_turbo_generate(self, prompt, num_retry = 5):
        # generate answer by gpt-3.5-turbo
        openai_key = os.environ.get("OPENAI_API_KEY")
        for _ in range(num_retry):
            try:
                r = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        
        return r.choices[0]['message']['content']
    
    def prompt_for_the_task(self):
        raise NotImplementedError
    
    def _generate_answer(self, prompt):
        raise NotImplementedError
    
    def generate_by_openai(self, prompt):
        # generate answer by openai
        pass
    
    def get_answer(self, contexts: List[ContextAndAnswer]):
        raise NotImplementedError
    
    def setup(self, ans: ContextAndAnswer):
        self.ans = ans
        self.dataset_type = ans.dataset_type
        self.mask_ratio = ans.mask_ratio

        # see if checkpoint exists
        file_path = os.path.join(self.save_path, f'answer_{self.model_type}_{self.task_name}_{self.dataset_type}_{self.mask_ratio}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                pickled_ans = pickle.load(f)
            logging.info(f'Loaded from {file_path}')
            print(f'Loaded from {file_path}')

            # update saved answers and questions to the latest
            self.ans.answer_of_contexts = pickled_ans.answer_of_contexts
            self.ans.questions = pickled_ans.questions

    def save_as_pickle(self):
        file_path = os.path.join(self.save_path, f'answer_{self.model_type}_{self.task_name}_{self.dataset_type}_{self.mask_ratio}.pkl')
        # save the ContextAndAnswer object as pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.ans, f)
        logging.info(f'Saved to {file_path}')
        print(f'Saved to {file_path}')
    
    def _result_output_path(self, file_path, dataset_type, model_type, context_id, context_type,):
        if context_type == 'no':
            return os.path.join(file_path, f"{dataset_type}_{model_type}_{context_id}_{context_type}.tsv")
        return os.path.join(file_path, f"{dataset_type}_{model_type}_{context_id}_{context_type}_{self.mask_ratio}.tsv")

class Evaluator:

    def __init__(self, metrics = ['bleu', 'meteor', 'rouge', ]):
        self._prepare_evaluation(metrics)
    
    def _prepare_evaluation(self, metrics: List[str]):
        # prepare evaluation
        # should support rouge, bleu, and other metrics?
        self.metrics = {}
        for metric in metrics:
            metric_ = evaluate.load(metric)
            self.metrics[metric] = metric_
        logging.info(f'Finished loading metrics: {self.metrics.keys()}')
        print(f'Finished loading metrics: {self.metrics.keys()}')
    
    def evaluate(self, predictions, references):
        # evaluate the answer
        # should support rouge, bleu, and other metrics?
        results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == 'bertscore':
                score = metric.compute(predictions=predictions, references=references, lang='en')
                score = {f'bertscore_{k}': np.mean(v) for k, v in score.items() if k in ['f1', 'precision', 'recall']}
            else:
                score = metric.compute(predictions=predictions, references=references)
            if metric_name == 'bleurt':
                score = {f'bleurt_{k}': v for k, v in score.items() if k in ['scores']}
            results.update(score)
        return results

class Summarisation(TaskManager):
    """
        This task is summarisation on the given context.
    """

    def __init__(self, task_name, model_type, save_path):
        super().__init__(task_name, model_type, save_path)

        self.summary_saved_path = os.path.join(self.save_path, task_name,)
        if not os.path.exists(self.summary_saved_path):
            os.makedirs(self.summary_saved_path)
    
    def prompt_for_the_task(self, context: ArxivContext):
        if self.model_type == "flan-t5-xxl":
            prompt = f"Summarize: {context.context}"
        elif 'vicuna' in self.model_type:
            prompt = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives professional answers to the user\'s request.\nUSER: \n----\n {context.context}\n\n----\n\n please summarize the above paragraph.\nASSISTANT:'
        elif self.model_instruct_tuned:
            prompt = f"{context.context}\n\n----\n\nSummarise the above content."
        elif not self.model_instruct_tuned:
            prompt = f"{context.context}\n\nTl;dr\n"
            # prompt = f"{context.context}\n\nThe summary:"
        return prompt

    def get_answer(self):
        ans = self.ans
        answer_of_contexts = ans.answer_of_contexts if ans.answer_of_contexts is not None else {}
        for context_type, contexts in ans.contexts_dict.items():
            answer_of_contexts[context_type] = []
            # if context_type not in answer_of_contexts:
            #     answer_of_contexts[context_type] = []
            # else:
            #     continue

            if self.model_type != "gpt-3.5-turbo":
                prompts = []
                out_files = []
            for context in contexts:
                summary_save_file = os.path.join(self.summary_saved_path, f"{ans.dataset_type}_{self.model_type}_{context.id}_{context_type}_{self.mask_ratio}.tsv")
                # summary_save_file = self._result_output_path(self.summary_saved_path, ans.dataset_type, self.model_type, context.id, context_type)
                if os.path.exists(summary_save_file):
                    pass
                else:
                    prompt = self.prompt_for_the_task(context)
                    if self.model_type == "gpt-3.5-turbo":
                        summary = self._generate_answer(prompt)
                        # save the summary
                        with open(summary_save_file, 'w',encoding='utf-8') as f:
                            f.write(summary)
                    else:
                        prompts.append(prompt)
                        out_files.append(summary_save_file)
            
            if self.model_type != "gpt-3.5-turbo" and len(prompts)!=0:
                # generate answers
                summaries = self._lm_answer_batch(prompts)
                for summary, summary_save_file in zip(summaries, out_files):
                    # save the summary
                    with open(summary_save_file, 'w',encoding='utf-8') as f:
                        f.write(summary)
                    print(f"Saved to {summary_save_file}")
            
            for context in contexts:
                summary_save_file = os.path.join(self.summary_saved_path, f"{ans.dataset_type}_{self.model_type}_{context.id}_{context_type}_{self.mask_ratio}.tsv")
                # summary_save_file = self._result_output_path(self.summary_saved_path, ans.dataset_type, self.model_type, context.id, context_type)
                # load the summary
                with open(summary_save_file, 'r',encoding='utf-8') as f:
                    summary = f.read()
                    if self.model_instruct_tuned:
                        if 'ASSISTANT:' in summary:
                            summary = summary.split('ASSISTANT:', 1)[1].strip()
                        else:
                            summary = summary
                    elif not self.model_instruct_tuned:
                        summary = summary.rsplit('\n', 1)[0].strip()

                answer_of_contexts[context_type].append(summary)
        ans.answer_of_contexts = answer_of_contexts
        self.ans = ans
        logging.info(f"Summarisation task is done.")
        return ans
    
    def evaluate(self, evaluator: Evaluator):
        # evaluate the summarisation task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        contexts = self.ans
        reference_context = contexts.reference_context
        reference_answer = contexts.answer_of_contexts[reference_context]
        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            answer = contexts.answer_of_contexts[context_type]
            reference_answer_ = reference_answer[:len(answer)]
            answers_ = []
            ref_ = []
            for a, r in zip(answer, reference_answer_):
                if isinstance(a, float) or isinstance(r, float):
                    continue
                answers_.append(a)
                ref_.append(r)
            performance[context_type] = evaluator.evaluate(predictions=answers_, references=ref_)
            
        self.ans.metrics = performance
        return performance
        
class MaskedTargetingQA(TaskManager):
    """
        This task is questions targeting on the masked sentences.
    """
    def __init__(self, task_name, model_type):
        super().__init__(task_name, model_type)
    
    def prompt_for_the_task(self):
        # prepare the prompt for the masked targeting QA task
        pass

    def get_answer(self, prompt):
        # generate answer for the given prompt
        pass

class QA(TaskManager):
    """
        This task conducts general QA on the given context.

        It first generate questions based on the given context.
        Then it generate answers for the questions given list of contexts.

        Note that the questions generated are shared across all contexts.
    """
    def __init__(self, task_name, model_type, save_path):
        super().__init__(task_name, model_type, save_path)

        self.question_saved_path = os.path.join(self.save_path, task_name,)
        if not os.path.exists(self.question_saved_path):
            os.makedirs(self.question_saved_path)
    
    def generate_questions(self, ans: ContextAndAnswer):
        # see if the questions are already generated
        if ans.questions is not None:
            return ans

        # generate questions based on the origin context
        origin_contexts = ans.contexts_dict[ans.reference_context]
        all_questions = []
        reference_answers = []
        for cont in origin_contexts:
            question_save_file = os.path.join(self.question_saved_path, f"{ans.dataset_type}_{cont.id}.tsv")
            if os.path.exists(question_save_file):
                pass
            else:
                # generate questions
                prompt = self.prompt_for_the_task(cont, task = "question_generation")
                questions = self._generate_answer(prompt)

                # save the questions
                with open(question_save_file, "w",encoding='utf-8') as f:
                    f.write(questions)

            # load the questions
            try:
                questions = pd.read_csv(question_save_file, sep = "\t", on_bad_lines='skip')
                questions_ = questions['Question'].tolist()
                answers = questions['Answer'].tolist()
            except Exception as e:
                print(f'File parse Error. {question_save_file}')
                questions_ = None
                answers = None

            all_questions.append(questions_)
            reference_answers.append(answers)
        
        ans.questions = all_questions
        if self.model_type == 'gpt-3.5-turbo':
            # other models need to generate answers from scratch
            ans.answer_of_contexts = {ans.reference_context: reference_answers}
        return ans

    def prompt_for_the_task(self, context: ArxivContext, task : str, questions: List[str] = None):
        assert task in ["question_generation", "answer_generation"], "task should be either question_generation or answer_generation"

        # prepare the prompt for question generation
        if task == "question_generation":
            prompt = f"Please generate a tsv file containing a list of question and answer based on the following given context. Remember, generate only the tsv content and nothing else. The two column names should be Question and Answer.\n\n---\n{context.context}"
        elif task == "answer_generation":
            questions = "\n".join([f"{idx+1}. {qus}" for idx, qus in enumerate(questions)])
            if self.model_type == 'flan-t5-xxl':
                prompt = f"Passage: {context.context}\n\nQuestions:\n{questions}\n\n Answers:"
            elif 'vicuna' in self.model_type:
                prompt = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives professional answers to the user\'s request.\nUSER: \n----\n {context.context}\n\n----\n\n please answer the following questions based on the given paragraph above.\n{questions}\n ASSISTANT:'
            elif not self.model_instruct_tuned:
                prompt = f"{context.context}\n\nGiven the above passage, they are asked to answer the following questions:\n{questions}\n\n the answer for each question is:\n\n"
            elif self.model_instruct_tuned:
                prompt = f"{context.context}\n\nGiven the above passage, answer the following questions:\n{questions}:"
                # prompt = f"Please generate a tsv file to answer the given questions based on the following given paragraph. Remember, generate only two columns for the question number and answers and nothing else. The column names should be Num and Answer.\n\n---Paragraph\n{context.context}\n\n---Questions\n{questions}"

        return prompt

    def get_answer(self):
        ans = self.ans
        answer_of_contexts = ans.answer_of_contexts if ans.answer_of_contexts is not None else {}
        logging.info(f"Answer generation task is started.")
        for context_type, contexts in ans.contexts_dict.items():
            answer_of_contexts[context_type] = []
            # if context_type not in answer_of_contexts:
            #     answer_of_contexts[context_type] = []
            # else:
            #     continue
            
            if self.model_type != 'gpt-3.5-turbo':
                prompts = []
                out_files = []
            for index, context in enumerate(contexts):
                if ans.questions[index] is None:
                    answer_of_contexts[context_type].append(None)
                    continue
                answer_save_file = self._result_output_path(self.question_saved_path, ans.dataset_type, self.model_type, context.id, context_type)

                if os.path.exists(answer_save_file):
                    pass
                else:
                    # generate questions
                    prompt = self.prompt_for_the_task(context, task = "answer_generation", questions = ans.questions[index])
                    if self.model_type == 'gpt-3.5-turbo':
                        # which means the model is running on OpenAI, so we do sequential generation
                        answers = self._generate_answer(prompt)

                        # save the questions
                        with open(answer_save_file, "w",encoding='utf-8') as f:
                            f.write(answers)
                    else:
                        # which means the model is running on real machine, so we do batch generation
                        prompts.append(prompt)
                        out_files.append(answer_save_file)
            
            if self.model_type != 'gpt-3.5-turbo' and len(prompts)!=0:
                outs = self._lm_answer_batch(prompts)
                for out_file, out in zip(out_files, outs):
                    with open(out_file, "w",encoding='utf-8') as f:
                        # we do not process the original output, we leave it to the post-processing below
                        f.write(out)
            
            for index, context in enumerate(contexts):
                if ans.questions[index] is None:
                    continue
                answer_save_file = self._result_output_path(self.question_saved_path, ans.dataset_type, self.model_type, context.id, context_type)
                # load the answers
                try:
                    with open(answer_save_file, "r",encoding='utf-8') as f:
                        answer = f.read()
                        if not self.model_instruct_tuned:
                            answers = [answer.rsplit("\n\n", 1)[0]]
                        elif self.model_instruct_tuned:
                            if 'ASSISTANT:' in answer:
                                answers = answer.split('ASSISTANT:', 1)[1].strip()
                            else:
                                answers = [answer]
                        # elif self.model_type == 'gpt-3.5-turbo':
                        #     answers = pd.read_csv(f, sep = "\t", on_bad_lines='skip')
                        #     answers = answers['Answer'].tolist()
                        #     assert len(answers) == len(ans.questions[index]), f"the number of answers {len(answers)} should be equal to the number of questions {len(ans.questions[index])}"

                except Exception as e:
                    print(f'Answer file parse Error. {answer_save_file}')
                    print(f'Error message: {e}')
                    answers = None

                answer_of_contexts[context_type].append(answers)
        ans.answer_of_contexts = answer_of_contexts
        self.ans = ans
        logging.info(f"Summarisation task is done.")
        return ans
    
    def evaluate(self, evaluator: Evaluator):
        # evaluate the summarisation task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        contexts = self.ans
        reference_context = contexts.reference_context

        reference_answer = contexts.answer_of_contexts[reference_context]

        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            performance[context_type] = {}

            # the answers here is a list of list of answers, should be flatten into a 1-D list
            # also remember to remove the None answers
            answers = contexts.answer_of_contexts[context_type]
            flatten_answer = []
            flatten_reference_answer = []
            for p_a, r_a in zip(answers, reference_answer):
                if p_a is None or r_a is None:
                    continue
                assert len(p_a) == len(r_a), f"the number of answers {len(p_a)} should be equal to the number of reference answers {len(r_a)}"
                for p, r in zip(p_a, r_a):
                    if isinstance(p, float) or isinstance(r, float):
                        continue
                    flatten_answer.append(p)
                    flatten_reference_answer.append(r)
            
            performance[context_type] = evaluator.evaluate(flatten_answer, flatten_reference_answer)
        
        self.ans.metrics = performance
        return performance
    
    def setup(self, ans):
        super().setup(ans)
        self.ans = self.generate_questions(ans)

class OriginalContextReconsutrction(TaskManager):

    def __init__(self, task_name, model_type, save_path):
        super().__init__(task_name, model_type, save_path)

        self.summary_saved_path = os.path.join(self.save_path, task_name,)
        if not os.path.exists(self.summary_saved_path):
            os.makedirs(self.summary_saved_path)

    def get_answer(self):
        ans = self.ans
        answer_of_contexts = ans.answer_of_contexts if ans.answer_of_contexts is not None else {}
        logging.info(f"Reconstruction task is started.")
        for context_type, contexts in ans.contexts_dict.items():
            if context_type == ans.reference_context:
                answer_of_contexts[context_type] = [context.context for context in contexts]
                continue
            if context_type not in answer_of_contexts:
                answer_of_contexts[context_type] = []
            else:
                continue

            if self.model_type != 'gpt-3.5-turbo':
                prompts = []
                out_files = []
            for context in contexts:
                summary_save_file = os.path.join(self.summary_saved_path, f"{ans.dataset_type}_{self.model_type}_{context.id}_{context_type}_{self.mask_ratio}.tsv")
                if os.path.exists(summary_save_file):
                    pass
                else:
                    prompt = self.prompt_for_the_task(context)
                    if self.model_type != 'gpt-3.5-turbo':
                        prompts.append(prompt)
                        out_files.append(summary_save_file)
                    else:
                        summary = self._generate_answer(prompt)

                        # save the summary
                        with open(summary_save_file, 'w',encoding='utf-8') as f:
                            f.write(summary)
            
            if self.model_type != 'gpt-3.5-turbo' and len(prompts)!=0:
                # generate the summaries in batch
                outs = self._lm_answer_batch(prompts)
                for out, out_file in zip(outs, out_files):
                    with open(out_file, 'w',encoding='utf-8') as f:
                        f.write(out)
            
            for context in contexts:
                summary_save_file = os.path.join(self.summary_saved_path, f"{ans.dataset_type}_{self.model_type}_{context.id}_{context_type}_{self.mask_ratio}.tsv")
                # load the summary
                with open(summary_save_file, 'r',encoding='utf-8') as f:
                    summary = f.read()
                    if not self.model_instruct_tuned:
                        summary = summary.rsplit("\n", 1)[0]
                    elif self.model_instruct_tuned:
                        if 'ASSISTANT:' in summary:
                            summary = summary.split('ASSISTANT:', 1)[1].strip()
                        else:
                            summary = summary
                answer_of_contexts[context_type].append(summary)
        ans.answer_of_contexts = answer_of_contexts
        self.ans = ans
        logging.info(f"Reconstruction task is done.")
        print(f"Reconstruction task is done.")
        return ans

    def prompt_for_the_task(self, context: ArxivContext):
        # prepare the prompt for original context reconstruction

        if 'vicuna' in self.model_type:
            prompt = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives professional answers to the user\'s request.\nUSER: \n----\n {context.context}\n\n----\n\nThere are some phrases omitted in the following paragraphs. Please infer the missing parts based on contextual clues, reconstruct and show me the original content.\nASSISTANT: The original content is as fellow: '
        elif self.model_instruct_tuned:
            prompt = f"There are some phrases omitted in the following paragraphs. Please infer the missing parts based on contextual clues and reconstruct and show me the original content. Remember, generate only the reconstruted paragraphs and nothing else.\n---\n{context.context}"
        elif not self.model_instruct_tuned:
            prompt = f"The noisy paragraph is as fellow: {context.context}\n\nThere are some phrases omitted above. The complete paragraphs are: "
        return prompt
    
    def evaluate(self, evaluator: Evaluator):
        # evaluate the reconstruction task
        # try to use BLEU, ROUGE, METEOR, and BERTScore
        # bleu, bertscore, meteor, rouge all implemented by huggingface.metrics

        contexts = self.ans
        reference_context = contexts.reference_context
        reference_answer = contexts.answer_of_contexts[reference_context]
        performance = {}
        for context_type in contexts.answer_of_contexts:
            if context_type == reference_context:
                continue
            answer = contexts.answer_of_contexts[context_type]
            reference_answer_ = reference_answer[:len(answer)]
            slice_ = min(len(reference_answer_), len(answer))
            performance[context_type] = evaluator.evaluate(answer[:slice_], reference_answer_[:slice_])
        self.ans.metrics = performance
        return performance

class ContinueConversation(OriginalContextReconsutrction):

    def prompt_for_the_task(self, context: ArxivContext,):
        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives professional answers to the user\'s request.\n{context.context}"
        return prompt
