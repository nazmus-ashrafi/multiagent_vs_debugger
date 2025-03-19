# Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

---

# Generation Command Example

Run the following command to generate results:

```sh
python main.py --dataset humaneval --signature --provider_and_model openai:gpt-3.5-turbo-0125 --flow basic --range full --output_path evaluation/basic_gpt35_turbo.jsonl
```

---

# Evaluation Command Example

To evaluate functional correctness, use:

```sh
python evaluation/evaluate_functional_correctness.py \
    --problem_file evaluation/data/HumanEval.jsonl.gz \
    --sample_file evaluation/basic_gpt35_turbo.jsonl
```

---

# Flow Options

Available flow options:

- `basic`
- `AC`
- `ACT`
- `debugger`
- `ac_debugger`
- `act_debugger`

---

# Dataset (`problem_file`) Options

Available datasets:

- `HumanEval.jsonl.gz`
- `HumanEvalPlus.jsonl.gz`

---

# LLM Options

## Hugging Face Endpoints

1. `HuggingFace:HuggingFaceH4/zephyr-7b-beta`
2. `HuggingFace:Qwen/Qwen2.5-Coder-32B-Instruct`
3. `HuggingFace:meta-llama/Meta-Llama-3-8B-Instruct`
4. `HuggingFace:Qwen/QwQ-32B-Preview`
5. `HuggingFace:microsoft/Phi-3.5-mini-instruct`
6. `HuggingFace:mistralai/Mistral-7B-Instruct-v0.2`

## Deepseek (Requires `--api_key`)

7. `deepseek:deepseek-chat`

## OpenAI

8. `openai:gpt-3.5-turbo-0125`
9. `openai:gpt-4o-mini`
10. `openai:gpt-4o`

## Anthropic (Requires `--api_key`)

11. `anthropic:claude-3-haiku-20240307`
12. `anthropic:claude-3-5-sonnet-20241022`
13. `anthropic:claude-3-5-haiku-20241022`

## Groq

14. `groq:llama-3.3-70b-versatile`
15. `groq:llama-3.1-8b-instant`
16. `groq:gemma2-9b-it`
17. `groq:mixtral-8x7b-32768`

## Vertex

18. `vertex:gemini-2.0-flash-exp`
19. `vertex:gemini-1.0-pro`

---

# Acknowledgement
Our implementation adapts code from [LDB](https://github.com/FloridSleeves/LLMDebugger) and prompt ideas from both [LDB](https://github.com/FloridSleeves/LLMDebugger) and [Self-collaboration Code Generation via ChatGPT](https://github.com/YihongDong/Self-collaboration-Code-Generation). We thank them for their high-quality open source code!