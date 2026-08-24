# Chronos-RAG: A Time-Aware Narrative RAG System
**An Exhaustive, Beginner-Friendly Guide to the Math, NLP, and AI behind the Project**

---

## 1. Introduction: The Problem with Standard RAG
Imagine you are reading *Game of Thrones*. Characters travel across continents, forge alliances, die, and sometimes come back to life. 

If you ask a standard AI question-answering system (known as **RAG** or **Retrieval-Augmented Generation**) a question like, *"Is Ned Stark alive?"*, the system might search the book, find a paragraph from chapter 1 where Ned is happily eating dinner, and confidently tell you, *"Yes, he is alive!"* 

Standard RAG systems fail at long narratives because they treat all text as equally valid, completely ignoring **time, narrative progression, and character state changes**. 

**Chronos-RAG** was built to solve this. We took a standard AI and gave it human-like memory capabilities—specifically the ability to understand that events happen in a timeline, that old memories fade unless they are important, and that character states change over time.

Here is the exhaustive step-by-step evolution of the project from its initial "Vanilla" state to the final AI architecture, explaining the NLP (Natural Language Processing) and mathematical concepts along the way.

---

## 2. The Baseline: Vanilla RAG (Configuration C0)
In standard RAG, the AI system takes a large book, chops it up into equal-sized paragraphs (called **chunks**), and converts each chunk into a list of numbers called a **Vector Embedding**. 

### The Math: Vector Embeddings & Cosine Similarity
An embedding is an AI's way of understanding meaning. It maps words or sentences into a high-dimensional mathematical space (often 768 dimensions). Words with similar meanings (e.g., "King" and "Queen") end up close together in this mathematical space.

When you ask a question, the AI embeds your question into a vector and calculates the **Cosine Similarity** between your question's vector and every chunk's vector in the book. 
The formula for cosine similarity between two vectors $\mathbf{A}$ and $\mathbf{B}$ is:

$$ \text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} $$

The chunks with the highest similarity score are given to the AI to answer the question.
*Problem:* Vanilla RAG grabs text purely based on keyword and meaning similarity, disregarding when the event happened.

---

## 3. Phase 1: Giving the AI a "Human" Brain

### Step 3.1: Semantic Chunking (Configuration C1)
Instead of chopping the book blindly every 500 words, Chronos-RAG uses **Semantic Chunking**. 
We look at the mathematical cosine similarity between sentence vectors. If the similarity between Sentence A and Sentence B suddenly drops below a threshold (e.g., 0.75), it means the topic has changed, so we split the chunk there. This ensures chunks contain complete thoughts.

### Step 3.2: Event Classification (Configuration C2)
Not all text is equally important. A description of a feast is less important than a character dying. 
We implemented a **Machine Learning Classifier** to tag sentences into categories (e.g., `DIALOGUE`, `ACTION`, `DISCOVERY`, `DEATH`).

*The AI Aspect:* We initially tried a heavy Transformer neural network, but it caused the laptop to overheat and deadlock. We optimized the system by replacing it with a highly efficient **Linear Probe (Logistic Regression)** built using Scikit-Learn. 
Logistic regression takes text features (TF-IDF vectors) and applies the sigmoid function to predict the probability of an event type:

$$ P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} $$

If the ML model's confidence drops below $0.45$, we use a fallback rule-based system (checking for specific keywords) to ensure reliability.

### Step 3.3: Temporal Decay — The Atkinson-Shiffrin Memory Model (Config C3/C4)
How do humans remember things? The **Atkinson-Shiffrin model** states we have:
1. **Working Memory:** Recent things (lasts seconds/minutes).
2. **Episodic Memory:** Specific events (decays over time).
3. **Long-Term Memory:** Highly important facts (never decays).

We applied this cognitive psychology model to the AI mathematically!
Every event is given an initial importance score $S_0$. As narrative time progresses (measured by a `days_gap`), the memory decays exponentially based on a decay rate $\lambda$:

$$ S(t) = S_0 \cdot e^{-\lambda \Delta t} $$

- **Working Memory** decays very fast (high $\lambda$).
- **Long-Term Memory** has $\lambda = 0$ (no decay).
If an event's score $S(t)$ drops below a threshold, the AI "forgets" it. This prevents the AI from getting confused by old, irrelevant details.

### Step 3.4: Knowledge Graphs & Semantic Deduplication (Config C5/C6)
If Jon Snow swings his sword 50 times in a book, the AI will remember 50 similar events. We use **Semantic Deduplication**: if two events have a cosine similarity above $0.92$, we merge them into one to save memory.

Furthermore, we extract explicit **Character States** (e.g., "Ned Stark -> DEAD") and store them in a **Knowledge Graph**. A knowledge graph is a network of nodes and edges connecting entities. By injecting these facts directly into the retrieved context, the AI always knows the *current* state of the world, curing the core flaw of Vanilla RAG.

---

## 4. Phase 2: Hybrid Retrieval Engine (Configuration C8)
Even with human-like memory, Vector Embeddings (Dense search) have a flaw: they are bad at exact keyword matches (e.g., finding the specific name "Waymar Royce"). 

To fix this, we implemented **Hybrid Search**:
1. **Dense Search (Vector Embeddings):** Good at finding conceptual meaning.
2. **Sparse Search (BM25):** Good at finding exact keywords. BM25 is a mathematical formula that calculates how rare and important a word is across all documents.

We combine their results using an algorithm called **Reciprocal Rank Fusion (RRF)**:

$$ RRF(d) = \frac{1}{k + \text{rank}_{\text{dense}}(d)} + \frac{1}{k + \text{rank}_{\text{sparse}}(d)} $$

This ensures the retrieved context has both the right *meaning* and the right *names*.

---

## 5. Phase 3: Hardware Optimization & Evaluation
### CPU Batch Processing
Running heavy Neural Networks (like Ollama) for embeddings caused thermal throttling on your machine (~90°C). We optimized the architecture by replacing the sequential Ollama server requests with **CPU Batch Encoding** using `sentence-transformers`. This processes memories in bulk directly on the CPU, drastically lowering thermal load and execution time.

### Generative Evaluation (LLM-as-a-Judge)
How do we know if Chronos-RAG actually works? We built an evaluation pipeline where a local LLM (`Llama 3`) acts as an automated judge. It scores the final generated answers on two metrics:
1. **Relevance:** Does it correctly answer the ground truth?
2. **Faithfulness:** Does the AI hallucinate, or does it stick strictly to the text?

**The Result:** The system achieved a near-perfect Faithfulness score. Furthermore, when tested on events that happened *after* the current reading bookmark, the AI correctly responded, *"I don't know."* This proves that Chronos-RAG is highly reliable; it refuses to hallucinate facts it hasn't "read" yet, a massive achievement for AI systems!

---

## 6. Phase 4: Limitations & Future Work

While Chronos-RAG successfully mitigates many flaws of standard LLM-based memory systems, it inherently introduces new algorithmic trade-offs that warrant future academic study.

### A. Non-Linear Narrative Distortion
The decay model assumes narrative time proceeds sequentially. Thus, non-linear narratives (e.g., extensive flashbacks, time jumps, or parallel POVs) distort the temporal distance $\Delta t$. While **Dynamic Decay Scaling** partially solves this by weighting emotional and positional salience to preserve flashbacks, true non-linear temporal modeling remains an open challenge.

### B. Graph Fragility & Coreference Resolution
The explicit State Extraction logic constructs a global knowledge graph, but it relies on exact-match Entity Recognition (NER). It is fragile against **coreference** (e.g., "The King died" instead of "Robert Baratheon died"). While we introduced **Uncertainty Tracking and State Rollbacks** to manage conflicting rumors and unverified rules, a deep coreference resolution pipeline (like SpaCy neural coref) is needed to build a truly bulletproof Knowledge Graph.

### C. The Thermal-Syntactic Tradeoff
To prevent hardware throttling (laptop thermals >90°C), we replaced heavy Transformer-based event classifiers with a Scikit-Learn `LogisticRegression` probe running on TF-IDF sparse features. While this executes orders of magnitude faster, TF-IDF drops structural syntactic context and word order. As a result, the linear probe struggles with complex literary devices like sarcasm or implied state transitions, creating a persistent tradeoff between execution speed and contextual depth.

### D. Bias in Generative Evaluation (LLM-as-a-Judge)
The generative performance is scored using an LLM-as-a-judge paradigm (`Llama 3`). It is well documented in NLP literature that LLMs exhibit a "self-preference bias" and can occasionally misalign on abstract reasoning tasks. While running the system against standardized benchmarks like **NarrativeQA** grounds the pipeline, future iterations should incorporate human-in-the-loop (HITL) annotations for true gold-standard metrics.

---

## Conclusion
From a baseline RAG system that mindlessly retrieved text, Chronos-RAG evolved into a cognitively inspired, multi-modal, and time-aware AI. By fusing **Machine Learning classifiers, mathematical decay modeling, knowledge graphs, and hybrid retrieval algorithms**, this project perfectly demonstrates how standard AI architectures can be augmented to truly "understand" and "remember" dynamic, evolving environments.
