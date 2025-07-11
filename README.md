# 🧠 LangChain Learning Tasks @ AlgoCas Internship

This repository contains hands-on tasks exploring the capabilities of **LangChain**, a powerful framework for building LLM-powered applications. Each task is organized in a Jupyter notebook or app and focuses on a key LangChain concept.

---

## 📁 Repository Structure

### 🔵 Week 3

#### 🔹 Task 1: LLMChain and Text Summarization
- ✅ Build a basic **LLMChain** using prompt templates.
- 📝 Implement **text summarization** workflows using LangChain.
- 📚 Learn prompt engineering and chain configuration fundamentals.

#### 🔹 Task 2: LangChain Tools (Search & Calculator)
- 🔍 Use built-in tools like **Search** and **Calculator**.
- 📄 Apply to structured files like **Excel** and **PDFs**.
- 🤖 Explore tool use within LangChain **agents**.

#### 🔹 Task 3: Vector Store + Retrieval QA
- 📦 Use **FAISS** to store document embeddings.
- ❓ Build a **Retrieval-Augmented Generation (RAG)** pipeline.
- 🧠 Query PDF content using **RetrievalQA**.

#### 🔹 Task 4: Memory Management
- 💬 Implement **conversation memory** (e.g., `ConversationBufferMemory`).
- 🔄 Handle context windows and persistent sessions.
- 🛠️ Compare memory types and how they affect chaining.

---

### 🟢 Week 4

#### 🔹 Task 1 & 3: LangChain Agents + APIs + UI
- 🧠 Create custom **LangChain agents** with tools.
- 🌐 Connect to external **APIs** (e.g., weather, news).
- 🖥️ Wrap into a **Gradio or Streamlit** app for real-time interaction.
- 📁 File: `Task 1&3 LangChain Agents Gradio & APIs.ipynb, streamlit_app.py`

#### 🔹 Task 2: LangChain with Chat Models
- 💬 Use `ChatPromptTemplate`.
- 🧪 Build conversational chains using **chat-specific models**.
- 📁 File: `Task 2 LangChain with chat models.ipynb`

#### 🔹 Task 4: Document Loaders & Indexing
- 📑 Load documents from **PDFs**, **web pages**, etc.
- 🧠 Use LangChain’s **indexing API** with **FAISS** + **RecordManager**.
- ❓ Perform multi-source **retrieval QA**.
- 📁 File: `Task 4 Document Loaders & indexing.ipynb`

---

### 🟠 Week 5

#### 🔹 Task 1 & 2: Multi-User Conversational System
- 👥 Build a **multi-user conversational system**.
- 🗃️ Handle **User ID management** and store conversation history **per user**.
- 📁 File: `Task 1&2.ipynb`

#### 🔹 Task 3: OCR Text Extraction System
- 🖼️ Build a system to **extract text from images**.
- 🤖 Integrate OCR pipelines for document understanding.
- 📁 File: `Task 3.ipynb`

#### 🔹 Task 4: LangChain Q&A on OCR Output
- ❓ Use text extracted with OCR as context for **LangChain Q&A**.
- 🧠 Build applications combining **vision + LLM reasoning**.
- 📁 File: `Task 4.ipynb`

---
### 🔴 Week 6

#### 🔹 Task 2.1: Qwen OCR on Colab
- 🖼️ Test **Qwen models** for OCR tasks in Colab.
- 📄 Generate text outputs from images efficiently.
- 📁 File: `Task 2.1 Qwen_OCR.ipynb`

#### 🔹 Task 2.2: LangChain Integration with Qwen OCR
- 🔗 Integrate **Qwen OCR outputs** with LangChain locally.
- 🧠 Build end-to-end pipelines combining **vision and LLM reasoning**.
- 📁 File: `Task 2.2 LangChain_Qwen.ipynb`


---

📁 **Folder Structure:**
- `Qwen/Output/`: Contains subfolders per image with:
  - Original image
  - JSON output
  - Used prompt

---

