# Awesome AI Papers

> A **bilingual (Chinese + English)** deep-reading project that writes "biographies" for the 100 papers that truly changed AI history.
>
> Distill-style depth × Wikipedia-style coverage × native-Chinese writing + native-English mirror × LLM-driven and scalable.

**English** | [中文](README.md)

**Sites**:

- 🇬🇧 English site: <https://awesome.papernotes.org/en/>
- 🇨🇳 Chinese site: <https://awesome.papernotes.org/>

**Progress**: **131** deep notes so far, covering 5 eras (1958 → 2026).

| Era | Time range | Representative works | Notes |
|------|----------|----------|--------|
| Era 1 · Foundations | 1957-2011 | Perceptron / Backprop / SVM / LSTM / LeNet / ImageNet | 14 |
| Era 2 · Deep Renaissance | 2012-2016 | AlexNet / Word2Vec / GAN / Attention / ResNet / AlphaGo | 24 |
| Era 3 · Attention Era | 2017-2019 | Transformer / AlphaZero / BERT / GPT / PPO / Mask R-CNN | 20 |
| Era 4 · Foundation Models | 2020-2022 | GPT-3 / DDPM / ViT / CLIP / AlphaFold2 / Stable Diffusion / CoT | 33 |
| Era 5 · Large Model Era | 2023-present | LLaMA / SAM / DPO / Mamba / Sora / o1 / DeepSeek-R1 | 40 |

---

## The 8-section deep template

Every deep note must contain:

1. **Historical Context** — Why did this paper appear at that exact moment? What was the field stuck on?
2. **Background and Motivation**
3. **Method in Detail** — original equations + a modern reproduction in code + detailed comparison tables
4. **Failed Baselines** — Which baselines lost at the time, and why? (Almost every other write-up skips this.)
5. **Key Experimental Numbers**
6. **Idea Lineage** — must include a Mermaid citation graph (predecessors → this paper → successors → misreadings)
7. **Modern Perspective** — N years later, which assumptions no longer hold up?
8. **Limitations / Related Work / Resources**

> ⚠️ **No scoring of top-tier papers.** Being included in the Top 100 is itself the highest praise — the frontmatter intentionally has no `awesome_score` field.

---

## All notes (grouped by era)

> Click any title on GitHub to read the corresponding English markdown. For the web version (TOC, Mermaid rendering, dark mode), visit <https://awesome.papernotes.org/en/>.

### 5 · Large Model Era (2023-present)

- **2026** · [Claude Fable 5 / Mythos 5: When Safeguards Split One Model into Two Products](docs_en/era5_genai_explosion/2026_claude_fable5.md)
- **2026** · [DeepSeek-V4: Making Million-Token Context an Architectural Problem](docs_en/era5_genai_explosion/2026_deepseek_v4.md)
- **2026** · [GLM-5: From Vibe Coding to Sustained Agentic Engineering](docs_en/era5_genai_explosion/2026_glm5.md)
- **2026** · [GPT-6 Astra: When a Highly Autonomous Agent Crossed the Critical Cyber Threshold](docs_en/era5_genai_explosion/2026_gpt6_astra.md)
- **2025** · [Claude 3.5/3.7 Sonnet - Turning Frontier Models into Controllable Engineering Collaborators](docs_en/era5_genai_explosion/2025_claude_sonnet.md)
- **2025** · [DeepSeek-R1 — How Pure Reinforcement Learning Taught an Open LLM to Reason](docs_en/era5_genai_explosion/2025_deepseek_r1.md)
- **2025** · [GPT-5: When Model Selection Became a System Capability](docs_en/era5_genai_explosion/2025_gpt5.md)
- **2025** · [Gemini 2.5 - When Thinking, Multimodality, Long Context, and Tools Become One Model Family](docs_en/era5_genai_explosion/2025_gemini25.md)
- **2025** · [Gemma 3 - Packing Multimodal Long Context into a Single-Accelerator Open-Weight Family](docs_en/era5_genai_explosion/2025_gemma3.md)
- **2025** · [Llama 4 - Open-Weight Experiments in Native Multimodality, Sparse Experts, and Ten-Million-Token Context](docs_en/era5_genai_explosion/2025_llama4.md)
- **2025** · [Qwen2.5 / Qwen3 - How Alibaba Turned Open LLMs into a Full-Stack Model Family](docs_en/era5_genai_explosion/2025_qwen3.md)
- **2025** · [The Kimi K2 Family: From Open Agentic Intelligence to a 3T-Class Frontier Model](docs_en/era5_genai_explosion/2025_kimi_k2.md)
- **2024** · [AlphaFold 3 — Unifying Biomolecular Complex Prediction with All-Atom Diffusion](docs_en/era5_genai_explosion/2024_alphafold3.md)
- **2024** · [DeepSeek-V2 / V3 - How MLA and MoE Pushed Open Models to the Frontier](docs_en/era5_genai_explosion/2024_deepseek_v3.md)
- **2024** · [GPT-4o: From a Three-Model Voice Pipeline to 320 ms Omni Interaction](docs_en/era5_genai_explosion/2024_gpt4o.md)
- **2024** · [Gemini 1.5 - Multimodal Understanding Across Million-Token Contexts](docs_en/era5_genai_explosion/2024_gemini15.md)
- **2024** · [Genie: Generative Interactive Environments](docs_en/era5_genai_explosion/2024_genie.md)
- **2024** · [Llama 3 Herd - An Engineering Blueprint for Open Frontier Models](docs_en/era5_genai_explosion/2024_llama3.md)
- **2024** · [OpenAI o1 -> o3/o4-mini - From Test-Time Compute to Agentic Reasoning](docs_en/era5_genai_explosion/2024_o1.md)
- **2024** · [Phi-4: Turning Data Quality into a Small-Model Reasoning System](docs_en/era5_genai_explosion/2024_phi4.md)
- **2024** · [SWE-bench + SWE-agent — When Coding Left the Function Sandbox](docs_en/era5_genai_explosion/2024_swe_bench.md)
- **2024** · [Sora Technical Report - Video Generation Models as World Simulators](docs_en/era5_genai_explosion/2024_sora.md)
- **2024** · [Stable Diffusion 3 / Rectified Flow — Moving Text-to-Image from U-Net Diffusion to Scalable MMDiT](docs_en/era5_genai_explosion/2024_stable_diffusion3.md)
- **2023** · [3DGS — Bringing NeRF-Quality Radiance Fields into Real-Time Interaction](docs_en/era5_genai_explosion/2023_3dgs.md)
- **2023** · [AudioLM - Turning Raw Audio into a Language Modeling Problem](docs_en/era5_genai_explosion/2023_audiolm.md)
- **2023** · [DINOv2 - Robust Visual Features without Supervision](docs_en/era5_genai_explosion/2023_dinov2.md)
- **2023** · [DPO — Aligning LLMs Directly from Preferences without a Reward Model or PPO](docs_en/era5_genai_explosion/2023_dpo.md)
- **2023** · [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](docs_en/era5_genai_explosion/2023_diffusion_policy.md)
- **2023** · [GPT-4 Technical Report - Capability Leap and the Black-Box Technical Report](docs_en/era5_genai_explosion/2023_gpt4.md)
- **2023** · [LLaMA — How Smaller Parameters + More Tokens Helped Open-Source LLMs Catch Up to GPT-3](docs_en/era5_genai_explosion/2023_llama.md)
- **2023** · [LLaVA - Turning GPT-4-Generated Visual Instructions into an Open Multimodal Assistant](docs_en/era5_genai_explosion/2023_llava.md)
- **2023** · [Llama 2: Open Foundation and Fine-Tuned Chat Models](docs_en/era5_genai_explosion/2023_llama2.md)
- **2023** · [Mamba — How Selective State Spaces Became the First Credible Transformer Challenger in a Decade](docs_en/era5_genai_explosion/2023_mamba.md)
- **2023** · [Mixtral 8x7B — Open-Weight LLMs Enter the Sparse Expert Era](docs_en/era5_genai_explosion/2023_mixtral.md)
- **2023** · [QLoRA — Bringing 65B LLM Fine-Tuning onto a Single 48GB GPU](docs_en/era5_genai_explosion/2023_qlora.md)
- **2023** · [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](docs_en/era5_genai_explosion/2023_rt2.md)
- **2023** · [SAM — One Prompt + 11M Images + 1B Masks: Turning Segmentation into a Foundation Model Problem](docs_en/era5_genai_explosion/2023_sam.md)
- **2023** · [Toolformer - Letting Language Models Teach Themselves When to Use Tools](docs_en/era5_genai_explosion/2023_toolformer.md)
- **2023** · [Tree of Thoughts — Deliberate Search as a Reasoning Interface for LLMs](docs_en/era5_genai_explosion/2023_tot.md)
- **2023** · [vLLM / PagedAttention — Rescuing LLM Serving from KV-Cache Fragmentation](docs_en/era5_genai_explosion/2023_vllm.md)

### 4 · Foundation Models (2020-2022)

- **2023** · [ControlNet — Plugging spatial control into frozen diffusion via zero-convolutions](docs_en/era4_foundation_models/2022_controlnet.md)
- **2022** · [Chinchilla — Proving All LLMs Were 'Undertrained' via Compute-Optimal Allocation](docs_en/era4_foundation_models/2022_chinchilla.md)
- **2022** · [Classifier-Free Diffusion Guidance — One Line of Code Removes the Bolt-On Classifier and Unifies Modern Text-to-Image](docs_en/era4_foundation_models/2022_cfg.md)
- **2022** · [CoT — Unlocking LLM Reasoning with 'Let's Think Step by Step'](docs_en/era4_foundation_models/2022_cot.md)
- **2022** · [Constitutional AI — Replacing Tens of Thousands of Human Harm Labels With a Constitution and AI Feedback](docs_en/era4_foundation_models/2022_constitutional_ai.md)
- **2022** · [DiT - When Diffusion Models Replaced the U-Net with a Transformer](docs_en/era4_foundation_models/2022_dit.md)
- **2022** · [DreamBooth — Implanting Any Subject Into a Text-to-Image Model With 3-5 Photos](docs_en/era4_foundation_models/2022_dreambooth.md)
- **2022** · [Flamingo: a Visual Language Model for Few-Shot Learning](docs_en/era4_foundation_models/2022_flamingo.md)
- **2022** · [FlashAttention: Faster Attention Without Dropping a Single Interaction](docs_en/era4_foundation_models/2022_flashattention.md)
- **2022** · [Imagen — Cascaded Text-to-Image Diffusion with Deep Language Understanding](docs_en/era4_foundation_models/2022_imagen.md)
- **2022** · [InstructGPT — Turning GPT-3 from a Continuator into an Obedient Assistant via RLHF](docs_en/era4_foundation_models/2022_instructgpt.md)
- **2022** · [MAE — Teaching ViT Self-Supervised Pretraining via 75% Masking](docs_en/era4_foundation_models/2022_mae.md)
- **2022** · [PaLM — Scaling Dense Language Models to 540B with Pathways](docs_en/era4_foundation_models/2022_palm.md)
- **2022** · [ReAct: Synergizing Reasoning and Acting in Language Models](docs_en/era4_foundation_models/2022_react.md)
- **2022** · [Stable Diffusion — Moving Diffusion into Latent Space so Consumer GPUs Can Generate Images](docs_en/era4_foundation_models/2022_stable_diffusion.md)
- **2022** · [Whisper - Turning 680k Hours of Weak Supervision into a General Speech Interface](docs_en/era4_foundation_models/2022_whisper.md)
- **2021** · [AlphaFold2 — Driving Protein Structure Prediction to Atomic Accuracy via Attention + Evoformer](docs_en/era4_foundation_models/2021_alphafold2.md)
- **2021** · [CLIP — Teaching Vision Models to Understand Language via 400M Image-Text Pairs](docs_en/era4_foundation_models/2021_clip.md)
- **2021** · [Codex — Evaluating Large Language Models Trained on Code](docs_en/era4_foundation_models/2021_codex.md)
- **2021** · [DALL-E — Recasting Text-to-Image Generation as Language Modeling](docs_en/era4_foundation_models/2021_dalle.md)
- **2021** · [LoRA — Slashing Large-Model Fine-tuning Cost by 99% via Low-Rank Matrices](docs_en/era4_foundation_models/2021_lora.md)
- **2021** · [Swin Transformer - Turning ViT into a General-Purpose Vision Backbone with Shifted Windows](docs_en/era4_foundation_models/2021_swin_transformer.md)
- **2020** · [DDPM — Crowning Diffusion as the King of Image Generation via Thousand-Step Denoising](docs_en/era4_foundation_models/2020_ddpm.md)
- **2020** · [DETR — Recasting Object Detection as Transformer Set Prediction](docs_en/era4_foundation_models/2020_detr.md)
- **2020** · [GPT-3 — When 175B Parameters Made Prompting the New Programming Paradigm](docs_en/era4_foundation_models/2020_gpt3.md)
- **2020** · [MoCo: Queues, Momentum Encoders, and the Moment Vision Self-Supervision Became Transferable](docs_en/era4_foundation_models/2020_moco.md)
- **2020** · [NeRF — Encoding a Scene into a Differentiable Radiance Field with One MLP](docs_en/era4_foundation_models/2020_nerf.md)
- **2020** · [RAG — Turning Wikipedia into Replaceable Memory for Generation](docs_en/era4_foundation_models/2020_rag.md)
- **2020** · [Scaling Laws for Neural Language Models](docs_en/era4_foundation_models/2020_scaling_laws.md)
- **2020** · [Score SDE — Unifying Score-Based and Diffusion Models through Stochastic Differential Equations](docs_en/era4_foundation_models/2020_score_sde.md)
- **2020** · [SimCLR — A Plain Contrastive Loss That Crowned Self-Supervised Vision on ImageNet Linear Eval](docs_en/era4_foundation_models/2020_simclr.md)
- **2020** · [ViT — Dethroning Convolution from Vision with Pure Transformer](docs_en/era4_foundation_models/2020_vit.md)
- **2020** · [wav2vec 2.0 - Speech Recognition After 53k Hours of Listening and 10 Minutes of Labels](docs_en/era4_foundation_models/2020_wav2vec2.md)

### 3 · Attention Era (2017-2019)

- **2019** · [EfficientNet — Redefining CNN Efficiency via Compound Scaling](docs_en/era3_attention/2019_efficientnet.md)
- **2019** · [GPT-2 — Announcing the LLM Era with Scale and Zero-shot](docs_en/era3_attention/2019_gpt2.md)
- **2019** · [RoBERTa — The Engineering Audit That Re-trained BERT Properly](docs_en/era3_attention/2019_roberta.md)
- **2019** · [T5 — Unifying All NLP Tasks as Text-to-Text](docs_en/era3_attention/2019_t5.md)
- **2018** · [BERT — Ushering NLP into the Pretraining Era via Masked Language Modeling](docs_en/era3_attention/2018_bert.md)
- **2018** · [ELMo — Bringing Contextual Embeddings Mainstream via BiLSTM Bidirectional LMs](docs_en/era3_attention/2018_elmo.md)
- **2018** · [GPT-1 — Igniting the Pre-training Revolution with Decoder-only Transformer](docs_en/era3_attention/2018_gpt1.md)
- **2018** · [Graph Attention Networks (GAT) — Attention as a Learnable Graph Edge](docs_en/era3_attention/2018_gat.md)
- **2018** · [Group Normalization - Freeing Normalization from Batch Size](docs_en/era3_attention/2018_group_norm.md)
- **2018** · [PGD Adversarial Training — Robustness as Min-Max Optimization](docs_en/era3_attention/2018_pgd.md)
- **2018** · [SE-Net — Channel Attention Crowning the ILSVRC 2017 Champion](docs_en/era3_attention/2018_senet.md)
- **2018** · [StyleGAN — Pushing GAN to Photorealistic Face Generation via Style Modulation](docs_en/era3_attention/2018_stylegan.md)
- **2017** · [AlphaZero — Erasing Human Go Knowledge from RL via Pure Self-Play](docs_en/era3_attention/2017_alphazero.md)
- **2017** · [CycleGAN — Unlocking Unpaired Image Translation via Cycle Consistency Loss](docs_en/era3_attention/2017_cyclegan.md)
- **2017** · [GCN — Founding Semi-supervised Node Classification and Graph Neural Networks](docs_en/era3_attention/2017_gcn.md)
- **2017** · [Mask R-CNN — Unifying Instance Segmentation by Adding One Branch to Faster R-CNN](docs_en/era3_attention/2017_mask_rcnn.md)
- **2017** · [MobileNet — Bringing Deep Learning to Mobile Devices via Depthwise Separable Convolutions](docs_en/era3_attention/2017_mobilenet.md)
- **2017** · [PPO — How Clipping Finally Made Policy Gradient Tunable and Usable](docs_en/era3_attention/2017_ppo.md)
- **2017** · [PointNet — Permutation-Invariant Deep Networks for Unordered Point Clouds](docs_en/era3_attention/2017_pointnet.md)
- **2017** · [Transformer — Burying Recurrence with Attention](docs_en/era3_attention/2017_transformer.md)

### 2 · Deep Renaissance (2012-2016)

- **2016** · [AlphaGo — Defeating the Human Go World Champion with MCTS + Deep Networks](docs_en/era2_deep_renaissance/2016_alphago.md)
- **2016** · [LayerNorm: Normalization Without a Batch](docs_en/era2_deep_renaissance/2016_layer_norm.md)
- **2016** · [YOLO — Turning Object Detection into a Single Real-Time Regression](docs_en/era2_deep_renaissance/2016_yolo.md)
- **2015** · [BatchNorm — Turning Training Stability into a Layer](docs_en/era2_deep_renaissance/2015_batchnorm.md)
- **2015** · [FCN - Turning Classification Networks into Pixel-Level Segmenters](docs_en/era2_deep_renaissance/2015_fcn.md)
- **2015** · [Faster R-CNN — Learning Region Proposals Inside the Detector](docs_en/era2_deep_renaissance/2015_faster_rcnn.md)
- **2015** · [He Init - The Starting Point That Kept ReLU Networks Alive](docs_en/era2_deep_renaissance/2015_he_init.md)
- **2015** · [Inception / GoogLeNet — Making CNNs Deeper by Making Them Wider](docs_en/era2_deep_renaissance/2015_inception.md)
- **2015** · [Knowledge Distillation — Pouring a Large Model's Dark Knowledge into a Small One](docs_en/era2_deep_renaissance/2015_knowledge_distillation.md)
- **2015** · [Nature DQN - The Atari Moment That Made Deep Reinforcement Learning Public](docs_en/era2_deep_renaissance/2015_dqn_nature.md)
- **2015** · [ResNet — How Deep Residual Learning Unlocked the 152-Layer Door](docs_en/era2_deep_renaissance/2015_resnet.md)
- **2015** · [U-Net — Turning Encoder-Decoders and Skip Connections into the Default Grammar of Medical Segmentation](docs_en/era2_deep_renaissance/2015_unet.md)
- **2014** · [Adam — Adaptive Moments for Stochastic Optimization](docs_en/era2_deep_renaissance/2014_adam.md)
- **2014** · [Adversarial Examples — Linearity, FGSM, and the Beginning of Modern Robustness](docs_en/era2_deep_renaissance/2014_adversarial_examples.md)
- **2014** · [Bahdanau Attention — Teaching Neural MT Where to Look](docs_en/era2_deep_renaissance/2014_attention.md)
- **2014** · [GAN — Adversarial Games that Taught Neural Networks to Forge](docs_en/era2_deep_renaissance/2014_gan.md)
- **2014** · [R-CNN — The ImageNet Feature Hierarchy That Rebooted Detection](docs_en/era2_deep_renaissance/2014_rcnn.md)
- **2014** · [Seq2Seq - Compress Any Sequence into a Vector, Then Decode It Back](docs_en/era2_deep_renaissance/2014_seq2seq.md)
- **2014** · [VGG — Pushing CNNs to 19 Layers with 3×3 Convolutions](docs_en/era2_deep_renaissance/2014_vgg.md)
- **2013** · [DQN — The First Deep RL Agent to Learn Atari from Pixels](docs_en/era2_deep_renaissance/2013_dqn.md)
- **2013** · [VAE — Turning Generative Modeling into a Tractable Variational Bound](docs_en/era2_deep_renaissance/2013_vae.md)
- **2013** · [Word2Vec - The Industrial Shortcut that Put Meaning into Vectors](docs_en/era2_deep_renaissance/2013_word2vec.md)
- **2012** · [AlexNet — Halving ImageNet Top-5 Error with GPU + ReLU + Dropout](docs_en/era2_deep_renaissance/2012_alexnet.md)
- **2012** · [Dropout — Randomly Turning Neurons Off to Stop Feature Co-adaptation](docs_en/era2_deep_renaissance/2012_dropout.md)

### 1 · Foundations (1957-2011)

- **2011** · [ReLU — How max(0, x) Turned Deep Networks from "Lab Toy" to "Industrial Cornerstone"](docs_en/era1_foundations/2011_relu.md)
- **2010** · [Glorot Init — Making Deep Networks Pass Signals Before They Learn](docs_en/era1_foundations/2010_glorot_init.md)
- **2009** · [ImageNet — How 15M Images Turned a 'Dataset' into the Fuse of the Deep Learning Revolution](docs_en/era1_foundations/2009_imagenet.md)
- **2008** · [t-SNE — The Visual Language of High-Dimensional Data Visualization](docs_en/era1_foundations/2008_tsne.md)
- **2006** · [Autoencoder — RBM Pretraining Wakes Neural Networks From Cold Storage](docs_en/era1_foundations/2006_autoencoder.md)
- **2006** · [DBN — How Layer-wise Greedy Pretraining Made Deep Networks Trainable for the First Time](docs_en/era1_foundations/2006_dbn.md)
- **2003** · [LDA — Promoting pLSA to a Generalizable Fully-Bayesian Topic Model with a Dirichlet Prior](docs_en/era1_foundations/2003_lda.md)
- **2001** · [Random Forests — Bagging + Feature Sampling that Crowned Decision Trees on the ML Throne](docs_en/era1_foundations/2001_random_forests.md)
- **1998** · [LeNet — Stitching Convolution, Pooling and Backprop into the First Industrial-Grade Deep Network](docs_en/era1_foundations/1998_lenet.md)
- **1997** · [LSTM — How Gating Made Recurrent Networks Remember Long Dependencies for the First Time](docs_en/era1_foundations/1997_lstm.md)
- **1992** · [SVM — How Max-Margin and the Kernel Trick Dominated Machine Learning for Two Decades](docs_en/era1_foundations/1992_svm.md)
- **1989** · [Universal Approximation — The Existence Theorem That Certified Neural Networks' Expressive Power](docs_en/era1_foundations/1989_universal_approximation.md)
- **1986** · [Backprop — Pulling Multi-layer Networks from 'Untrainable' into the Optimizable World via the Chain Rule](docs_en/era1_foundations/1986_backprop.md)
- **1958** · [Perceptron — How the First Hardware Neuron That Learns from Data Sparked AI as a Discipline](docs_en/era1_foundations/1958_perceptron.md)

---

## Repository layout

```
awesome-papers/
├── README.md
├── README_en.md
├── mkdocs_zh.yml
├── mkdocs_en.yml
├── docs_zh/                              # Chinese site source (1:1 mirror)
│   ├── index.md
│   ├── era1_foundations/               # 14 notes — 1957-2011 Foundations
│   │   ├── 1958_perceptron.md                 #   Perceptron — first hardware neuron that could learn from data
│   │   ├── 1986_backprop.md                   #   Backprop — chain rule turns deep nets from 'untrainable' into optimizable
│   │   └── ...                                #   (LSTM / LeNet / SVM / ImageNet / ReLU ...)
│   ├── era2_deep_renaissance/          # 24 notes — 2012-2016 Deep Renaissance
│   │   ├── 2012_alexnet.md                    #   AlexNet — GPU + ReLU + Dropout halve ImageNet Top-5 error
│   │   ├── 2015_resnet.md                     #   ResNet — deep residual learning unlocks 152-layer networks
│   │   └── ...                                #   (VAE / GAN / Word2Vec / Adam / AlphaGo ...)
│   ├── era3_attention/                 # 20 notes — 2017-2019 Attention Era
│   │   ├── 2017_transformer.md                #   Transformer — attention buries the recurrent neural network
│   │   ├── 2018_bert.md                       #   BERT — masked language modeling brings pretraining to all of NLP
│   │   └── ...                                #   (GPT-1/2 / PPO / Mask R-CNN / StyleGAN ...)
│   ├── era4_foundation_models/         # 33 notes — 2020-2022 Foundation Models
│   │   ├── 2020_gpt3.md                       #   GPT-3 — at 175B, prompting becomes the new programming paradigm
│   │   ├── 2021_clip.md                       #   CLIP — 400M image–text pairs teach vision models to read language
│   │   └── ...                                #   (DDPM / ViT / AlphaFold2 / Stable Diffusion / CoT ...)
│   ├── era5_genai_explosion/           # 40 notes — 2023-present Large Model Era
│   │   ├── 2023_llama.md                      #   LLaMA — smaller params + more tokens let open-source LLMs match GPT-3
│   │   ├── 2025_deepseek_r1.md                #   DeepSeek-R1 — pure RL teaches an open-source LLM to reason
│   │   └── ...                                #   (SAM / DPO / Mamba / Sora / o1 / Claude / Qwen3 ...)
├── docs_en/                              # English site source (one folder per era)
└── shared/<slug>/lineage.mmd             # Mermaid citation-graph sources shared by zh/en notes
```

---

## License

CC-BY-NC 4.0
