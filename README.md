# Qwen2.5-3B Reinforcement Learning Fine-Tuning with Unsloth and GRPO

This repository contains a complete reinforcement learning fine-tuning pipeline for the Qwen2.5-3B-Instruct language model. It leverages the high-performance capabilities of Unsloth to enable fast, memory-efficient training, and integrates with Hugging Face's `trl` library using the Generalized Reinforcement Policy Optimization (GRPO) trainer.

The goal of this project is to enhance the reasoning capabilities of the Qwen2.5 model through reward-driven learning, using structured answers and correctness as key criteria. The GSM8K dataset is used as the benchmark for supervised reinforcement fine-tuning.

## Why This Project

Large language models often produce answers that are either unstructured or incorrect when prompted with multi-step reasoning tasks. This project addresses that limitation by:

- Structuring outputs using XML-based reasoning and answer format
- Rewarding models for generating logically coherent and syntactically valid responses
- Applying fine-tuning using GRPO, optimized for multiple reward types

## Project Highlights

- **Model:** Qwen/Qwen2.5-3B-Instruct
- **Training Technique:** Reinforcement learning via GRPO
- **Optimization Strategy:** LoRA (Low-Rank Adaptation) + BitsAndBytes 4-bit quantization
- **Training Speedup:** Achieved through Unsloth's patching of vLLM and XFormers backend
- **Dataset:** GSM8K (Grade School Math) for language reasoning and numerical tasks
