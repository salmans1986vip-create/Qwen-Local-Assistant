
import json
import uuid
from pathlib import Path
import pymupdf
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# تقسيم صفحات PDF إلى مقاطع مع حفظ رقم الصفحة
def extract_pdf_chunks(file_bytes):
    chunks = []
    chunk_size = 900
    overlap = 150
    step = chunk_size - overlap

    with pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    ) as doc:
        for page_number, page in enumerate(doc, start=1):
            
            text = " ".join(page.get_text().split())    

            for start in range(0, len(text), step):
                chunk = text[start:start + chunk_size].strip()

                if chunk:
                    chunks.append({
                        "page": page_number,
                        "text": chunk
                    })

    return chunks

# إنشاء فهرس دلالي للمحاضرة
@st.cache_data(show_spinner="جاري فهرسة ملف PDF...", max_entries=3)
def build_pdf_index(file_bytes):
    chunks = extract_pdf_chunks(file_bytes)

    if not chunks:
        return [], []

    embeddings = OllamaEmbeddings(model="bge-m3")

    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    return chunks, vectors

# البحث عن المقاطع الأكثر ارتباطا بالسؤال
def search_pdf(question, chunks, vectors, top_k=6):
    from math import sqrt

    if not chunks:
        return []

    named_files = {
        c["file"].lower()
        for c in chunks
        if c.get("file") and c["file"].lower() in question.lower()
    }

    if named_files:
        kept_pairs = [
            (c, v) for c, v in zip(chunks, vectors)
            if c.get("file", "").lower() in named_files
        ]
        chunks, vectors = zip(*kept_pairs)
    embeddings = OllamaEmbeddings(model="bge-m3")
    query_vector = embeddings.embed_query(question)

    query_norm = sqrt(sum(v * v for v in query_vector)) or 1.0
    results = []

    for chunk, vector in zip(chunks, vectors):
        vector_norm = sqrt(sum(v * v for v in vector)) or 1.0
        score = sum(
            a * b for a, b in zip(query_vector, vector)
        ) / (query_norm * vector_norm)

        results.append((score, chunk))

    results.sort(key=lambda item: item[0], reverse=True)

    return [chunk for score, chunk in results[:top_k]]
# 1. إعداد الصفحة
st.set_page_config(
    page_title="مساعد Qwen المحلي",
    page_icon="🤖",
    layout="wide"
)


# 2. تنسيق اللغة العربية
st.markdown("""
<style>
.stApp {
    direction: rtl;
}
[data-testid="stChatMessage"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stChatMessageContent"] p {
    direction: rtl;
    text-align: right;
    unicode-bidi: isolate;
    line-height: 1.9;
    font-size: 18px;
}
[data-testid="stChatInput"] textarea {
    direction: rtl;
    text-align: right;
}

/* عرض الأكواد من اليسار إلى اليمين */
[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] pre code,
[data-testid="stCodeBlock"] {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: isolate !important;
}
</style>
""", unsafe_allow_html=True)

# 3. ملفات حفظ المحادثات
folder = Path(__file__).parent
store_file = folder / "qwen_chats.json"
old_file = folder / "persistent_history.json"

# 4. حفظ جميع المحادثات
def save_chats():
    store_file.write_text(
        json.dumps(
            st.session_state.data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

# 5. تحميل المحادثات عند التشغيل
if "data" not in st.session_state:
    if store_file.exists():
        try:
            data = json.loads(store_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            st.error("تعذر قراءة ملف المحادثات")
            st.stop()
    else:
        old_messages = []
        if old_file.exists():
            try:
                old_messages = json.loads(
                    old_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                st.error("تعذر قراءة الذاكرة السابقة")
                st.stop()

        data = {
            "active": "original",
            "chats": {
                "original": {
                    "title": "المحادثة السابقة",
                    "messages": old_messages
                }
            }
        }

    st.session_state.data = data
    save_chats()

# 6. القائمة الجانبية
st.sidebar.title("💬 المحادثات")

if st.sidebar.button("➕ محادثة جديدة", use_container_width=True):
    chat_id = uuid.uuid4().hex

    st.session_state.data["chats"][chat_id] = {
        "title": "محادثة جديدة",
        "messages": []
    }
    st.session_state.data["active"] = chat_id
    save_chats()
    st.rerun()

st.sidebar.subheader("المحادثات السابقة")

for chat_id, chat in reversed(
    list(st.session_state.data["chats"].items())
):
    if st.sidebar.button(
        chat["title"],
        key=chat_id,
        use_container_width=True
    ):
        st.session_state.data["active"] = chat_id
        save_chats()
        st.rerun()

# 7. اختيار المحادثة الحالية
active_id = st.session_state.data["active"]
current_chat = st.session_state.data["chats"][active_id]
messages = current_chat["messages"]

st.title("🤖 مساعد Qwen المحلي")
st.caption("مساعد عربي يعمل محليًا مع ذاكرة دائمة")
st.subheader(current_chat["title"])

# 8. إعداد نموذج Qwen
@st.cache_resource
def create_chain():


   
    model = ChatOllama(
        model="qwen3.5:9b-q4_K_M",
        reasoning=False,
        num_predict=1536,
        num_ctx=8192
    )
       
    prompt = ChatPromptTemplate.from_messages([
        

        ("system", "أنت مساعد تعليمي ذكي باللغة العربية. أجب بوضوح عن سؤال المستخدم. لا تكتب أكواد Python أو أمثلة برمجية إلا إذا طلبها المستخدم صراحة. عند الإجابة من ملفات PDF التزم بالنصوص المقدمة، ولا تنسب معلومات أو أكواد إلى صفحات لا تحتوي عليها. إذا لم تتوفر أدلة كافية فاذكر ذلك."),        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    return prompt | model | StrOutputParser()

chain = create_chain()

# رفع ملف PDF واستخراج النص
pdf_text = ""

pdf_chunks = []
pdf_vectors = []


uploaded_pdfs = st.file_uploader(
    "رفع ملفات PDF للمساعد",
    type=["pdf"],
    accept_multiple_files=True
)



if uploaded_pdfs:
    for uploaded_pdf in uploaded_pdfs:
        try:
            file_bytes = uploaded_pdf.getvalue()

            # قراءة صفحات الملف
            with pymupdf.open(
                stream=file_bytes,
                filetype="pdf"
            ) as doc:
                pages = [page.get_text() for page in doc]

            file_text = "\n\n".join(pages)

            if not file_text.strip():
                st.warning(
                    f"لا يوجد نص قابل للاستخراج من {uploaded_pdf.name}"
                )
                continue

            # فهرسة الملف باستخدام BGE-M3
            new_chunks, new_vectors = build_pdf_index(file_bytes)

            # تسجيل اسم الملف لكل مقطع
            for chunk in new_chunks:
                chunk["file"] = uploaded_pdf.name

            # جمع مقاطع جميع الملفات
            pdf_chunks.extend(new_chunks)
            pdf_vectors.extend(new_vectors)

            pdf_text += (
                f"\n\nاسم الملف: {uploaded_pdf.name}\n"
                f"{file_text}"
            )

            st.success(
                f"تمت قراءة وفهرسة {uploaded_pdf.name} بنجاح"
            )

            # معاينة الصفحة الثانية
            with st.expander(
                f"معاينة الصفحة الثانية - {uploaded_pdf.name}"
            ):
                if len(pages) >= 2:
                    lines = [
                        line.strip()
                        for line in pages[1].splitlines()
                        if line.strip()
                    ]

                    if lines:
                        st.text("\n".join(lines[:3]))
                    else:
                        st.warning("لا يوجد نص في الصفحة الثانية")
                else:
                    st.warning("الملف أقل من صفحتين")

        except Exception as error:
            st.error(
                f"خطأ في قراءة {uploaded_pdf.name}: {error}"
            )
# 9. عرض رسائل المحادثة الحالية
for message in messages:
    role = "user" if message["role"] == "human" else "assistant"

    with st.chat_message(role):
        st.write(message["content"])

        if role == "assistant" and message.get("sources"):
            with st.expander("المصادر المسترجعة لهذا السؤال"):
                for item in message["sources"]:
                    st.write(
                        f"الملف: {item.get('file', 'غير معروف')} | "
                        f"الصفحة: {item.get('page', 'غير معروفة')}"
                    )
                    st.text(item.get("text", "")[:350])
# 10. استقبال سؤال جديد
question = st.chat_input("اكتب سؤالك هنا...")

if question:
    with st.chat_message("user"):
        st.write(question)

    history = []
    for message in messages:
        if message["role"] == "human":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "ai":
            history.append(AIMessage(content=message["content"]))

    with st.chat_message("assistant"):
        with st.spinner("جاري إعداد الإجابة..."):
            try:
                rag_question = question

                

                if pdf_chunks and pdf_vectors:
                    
                    
                        parts = question.split("ثم", 1)

                        if len(parts) == 2:
                            first_chunks = search_pdf(
                                parts[0].strip(), pdf_chunks, pdf_vectors, top_k=3
                            )
                            second_chunks = search_pdf(
                                parts[1].strip(), pdf_chunks, pdf_vectors, top_k=3
                            )
                            relevant_chunks = first_chunks + [
                                item for item in second_chunks
                                if item not in first_chunks
                            ]
                        else:
                            relevant_chunks = search_pdf(
                                question, pdf_chunks, pdf_vectors, top_k=6
                            )

                        with st.expander("المصادر المسترجعة لهذا السؤال"):
                            for item in relevant_chunks:
                                st.write(
                                    f"الملف: {item.get('file', 'غير معروف')} | "
                                    f"الصفحة: {item.get('page', 'غير معروفة')}"
                                )
                                st.text(item.get("text", "")[:350])

                        context = "\n\n".join(
                            f"[الملف: {item.get('file', 'غير معروف')} | الصفحة: {item.get('page', 'غير معروفة')}]\n{item['text']}"
                            for item in relevant_chunks
                        )

                        rag_question = (
                            "أجب اعتمادا على المقاطع التالية من ملف PDF فقط. "
                            
    "اذكر اسم الملف ورقم الصفحة بجانب كل معلومة مستخرجة."
                            "إذا لم تجد الإجابة في المقاطع فلا تخمن.\n\n"
                            
                    "لا تعتبر عناوين المحاضرة أو أهدافها أو فهرسها دليلاً على معلومة تقنية.\n"
                    "استشهد فقط بالصفحات التي تحتوي على شرح يدعم المعلومة فعلاً.\n"
                    "إذا كانت الأدلة لا تكفي للإجابة الكاملة، اذكر المعلومات المدعومة فقط ووضح ما ينقص.\n"
                    "لا تضف أمثلة برمجية من عندك إلا إذا طلبها المستخدم.\n\n"
                            f"مقاطع الملف:\n{context}\n\n"
                            f"سؤال المستخدم: {question}"
                        )

                answer = chain.invoke({
                    "question": rag_question,
                    "history": history
                })

            except Exception as error:
                st.error(f"تعذر توليد الإجابة: {error}")
                st.stop()

        st.write(answer)

        if pdf_chunks and pdf_vectors:
            with st.expander("المصادر المسترجعة لهذا السؤال"):
                for item in relevant_chunks:
                    st.write(
                        f"الملف: {item.get('file', 'غير معروف')} | "
                        f"الصفحة: {item.get('page', 'غير معروفة')}"
                    )
                    st.text(item.get("text", "")[:350])
    # تحديث المحادثة
    messages.extend([
        {"role": "human", "content": question},
       
        {"role": "ai", "content": answer, "sources": relevant_chunks if pdf_chunks and pdf_vectors else []}
    ])

    # تسمية المحادثة من السؤال الأول
    if current_chat["title"] == "محادثة جديدة":
        current_chat["title"] = question[:35]

    save_chats()
    st.rerun()