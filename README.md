# 🗺️ TravelMind: Transforming Travel Through Intelligent Reasoning

## Overview

TravelMind is a comprehensive AI travel planning system that addresses the challenges of personalized, intelligent travel assistance through advanced fine-tuned language models. Built on Qwen3 with multi-stage RLHF training and sophisticated RAG systems, TravelMind provides localized travel planning capabilities that can run efficiently on consumer hardware.

### 🎯 Key Problem Solutions

**Personalization Gap**: Traditional travel planning tools provide generic recommendations that don't account for individual preferences, constraints, or cultural considerations. TravelMind addresses this through:

- Multi-stage RLHF training (PPO, DPO, GRPO) to align responses with human travel planning preferences
- Contextual recommendation engine that considers budget, time, accessibility, and personal interests
- Cultural sensitivity training for region-appropriate travel advice

**Information Fragmentation**: Travel planning requires synthesizing information from multiple disparate sources. TravelMind solves this via:

- Advanced RAG systems (Traditional, Self-RAG, MemWalker-RAG) for knowledge integration
- Tool calling framework for real-time API integration
- Multi-modal data processing for PDFs, web content, and structured data

**Local Deployment Requirements**: Privacy-conscious users and regions with limited connectivity need offline capabilities. TravelMind provides:

- Lightweight model architecture (0.5B-3B parameters) optimized for consumer hardware
- Local knowledge base with travel destination information
- Efficient inference pipeline supporting CPU and single-GPU deployment

## 🏗️ Architecture Overview

### Core Components

#### **1. Language Model Foundation**

- **Base Models**: Qwen3 series (0.6B, 1.5B, 3B parameters)
- **Fine-tuning**: LoRA-based parameter-efficient training
- **Optimization**: 4-bit quantization support for memory efficiency

#### **2. Multi-Stage RLHF Pipeline**

**Supervised Fine-Tuning (SFT)**

- Dataset: Travel-QA specialized dataset for domain adaptation
- Training: Custom SFTTrainer with DeepSpeed integration
- Focus: Travel domain knowledge and conversational patterns

**PPO (Proximal Policy Optimization)**

- Reward Model: OpenAssistant/reward-model-deberta-v3-large-v2
- Application: Route optimization learning human preferences for travel sequences
- Benefits: Improves recommendation quality through reward-based learning

**DPO (Direct Preference Optimization)**

- Dataset: Human-Like-DPO adapted for travel domain
- Application: Direct alignment with human travel planning preferences
- Benefits: Eliminates need for explicit reward modeling while maintaining alignment

**GRPO (Group Relative Policy Optimization)**

- Application: Group-based preference learning for diverse travel scenarios
- Benefits: Handles multiple user types and cultural preferences simultaneously

#### **3. Advanced RAG Systems**

**Traditional RAG**

- Implementation: BM25 + vector similarity search
- Knowledge Base: Travel destination PDFs and web content
- Integration: LangChain-based pipeline for document processing

**Self-RAG**

- Adaptive retrieval mechanism with on-demand information gathering
- Reflection token system for content quality assessment
- Three-step execution: retrieval decision → multi-passage processing → quality evaluation

**MemWalker-RAG**

- Hierarchical memory tree construction for long-document processing
- Interactive navigation with LLM-guided path selection
- Working memory maintenance for error recovery and reasoning

#### **4. Tool Integration Framework**

**Supported APIs** (Framework Ready):

- Google Search API for real-time information
- Weather API for destination conditions
- Hotel booking API integration
- Flight/transportation booking APIs
- Shortest path calculation APIs

**Tool Architecture**:

- `ToolDispatcher`: Intelligent tool selection and orchestration
- `ToolExecutor`: API call execution and response processing
- `PromptTemplate`: Context-aware tool calling prompt engineering

## 📊 Training Data Sources

### Primary Datasets

- **Travel-QA Dataset**: Specialized Q&A pairs for travel domain SFT
- **CrossWOZ Dataset**: Chinese dialogue dataset for conversational context understanding
- **Human-Like-DPO Dataset**: Preference pairs adapted for travel alignment training

### Knowledge Sources

- **PDF Documents**: Comprehensive travel guides and destination information
- **Web Content**: Curated travel websites and booking platforms
- **Structured Data**: Hotel, flight, and attraction databases

## 🎯 Specific RLHF Applications

### Route Optimization

- **Learning Objective**: Human preferences for travel sequences, timing, and destination priorities
- **Training Data**: Route preference pairs with human feedback
- **Outcome**: Personalized itinerary generation that balances efficiency with enjoyment

### Contextual Recommendations

- **Learning Objective**: Alignment with user constraints (budget, time, accessibility needs)
- **Training Data**: Recommendation scenarios with constraint satisfaction feedback
- **Outcome**: Practical suggestions that respect user limitations

### Safety & Practicality Focus

- **Learning Objective**: Preference for safe, feasible travel plans over theoretically optimal but impractical routes
- **Training Data**: Safety-focused travel scenarios with risk assessment
- **Outcome**: Responsible travel planning with risk mitigation

### Cultural Sensitivity

- **Learning Objective**: Appropriate recommendations based on cultural and regional preferences
- **Training Data**: Cross-cultural travel scenarios with cultural appropriateness feedback
- **Outcome**: Culturally aware travel suggestions and etiquette guidance

## 🚀 Getting Started

### Quick Installation

```bash
# Clone repository
git clone https://github.com/1998frankchen/TravelMind.git
cd TravelMind

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run basic RAG system
python main.py --function use_rag

# Use RAG dispatcher with Self-RAG
python main.py --function rag_dispatcher --rag_type self_rag

# Launch web interface
python main.py --function use_rag_web_demo

# Start training pipeline
python main.py --function train
```

For detailed setup instructions, see [MODEL_DATA_SETUP.md](MODEL_DATA_SETUP.md).

## 📈 Model Performance

### Hardware Requirements

- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: RTX3090 or equivalent, 32GB RAM
- **Supported**: CPU inference available with longer response times

### Model Variants

- **Qwen3-0.6B**: Lightweight deployment, 2GB memory footprint
- **Qwen3-1.5B**: Balanced performance, 6GB memory footprint
- **Qwen3-3B**: High-quality responses, 12GB memory footprint

## 🔧 Advanced Features

### Multi-Agent Architecture

- Specialized agents for different travel planning aspects
- Coordinated multi-turn conversations
- Context management across planning sessions

### Evaluation Metrics

- ROUGE scores for response quality
- BLEU scores for translation accuracy
- BERTScore for semantic similarity
- Custom travel-specific metrics

### Extension Points

- Custom tool integration via API framework
- New RAG method implementation
- Additional RLHF training stages
- Domain-specific model fine-tuning

## 🛠️ Development Status

### ✅ Fully Implemented

- Multi-stage RLHF training pipeline (SFT, PPO, DPO, GRPO)
- Advanced RAG systems (Traditional, Self-RAG, MemWalker)
- LoRA-based fine-tuning framework
- Web-based user interface
- Model evaluation and metrics

### 🚧 In Development

- Real-time API integrations (Google Search, Weather, Booking)
- Enhanced cultural sensitivity training data
- Multi-language support expansion
- Mobile deployment optimization

### 🔮 Planned Features

- Voice interface integration
- Offline map integration
- Real-time booking capabilities
- Advanced multimodal support (images, maps)

## 🤝 Contributing

We welcome contributions! Areas where help is particularly valuable:

- Travel-specific training data curation
- API integration implementations
- Cultural sensitivity improvements
- Performance optimizations
- Documentation enhancements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/1998frankchen/TravelMind/issues)
- **Discussions**: [GitHub Discussions](https://github.com/1998frankchen/TravelMind/discussions)
- **Documentation**: See `docs/` directory for detailed guides

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon excellent work from the research community:

- [Qwen Team](https://github.com/QwenLM/Qwen3) for the foundation models
- [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) for the reward model
- [LangChain](https://github.com/langchain-ai/langchain) for RAG infrastructure
- [Hugging Face](https://huggingface.co/) for model hosting and tools

Special thanks to all contributors and the open-source community for making this project possible.

---

**TravelMind**: Where artificial intelligence meets wanderlust. Plan smarter, travel better. 🌍✈️
