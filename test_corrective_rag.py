import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load API keys
api_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "API.json")
with open(api_file_path, "r") as f:
    apis = json.load(f)
    

# Set environment variables
os.environ["OPENAI_API_KEY"] = apis["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = apis["TAVILY_API_KEY"]

# Import necessary modules
from searchers.corrective_rag import CorrectiveRAGSearcher
from chat_models.openai_chat_model import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def test_corrective_rag():
    """Test function for Corrective RAG."""
    print("🚀 Starting Corrective RAG Test...")
    
    try:
        # Initialize components
        print("📋 Initializing components...")
        
        # LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ LLM initialized (using GPT-4o Mini - most cost-effective)")
        
        # Embeddings
        embeddings = OpenAIEmbeddings()
        print("✅ Embeddings initialized")
        
        # Web search tool
        web_search_tool = TavilySearchResults(k=3)
        print("✅ Web search tool initialized")
        
        # FAISS index path and corpus path
        faiss_index_path = "data/faiss_index"
        corpus_path = "data/multihoprag_corpus.txt"
        
        print(f"📂 Using corpus: {corpus_path}")
        print(f"💾 FAISS index path: {faiss_index_path}")
        
        # Initialize CorrectiveRAGSearcher
        print("🔧 Initializing Corrective RAG Searcher...")
        crag_searcher = CorrectiveRAGSearcher(
            faiss_index_path=faiss_index_path,
            corpus_path=corpus_path,
            llm=llm,
            embeddings=embeddings,
            k=3,
            web_search_tool=web_search_tool
        )
        print("✅ Corrective RAG Searcher initialized successfully!")
        
        # Test questions
        test_questions = [
            "What are the types of agent memory?",
            "How does AlphaCodium work?",
            "What is machine learning?"
        ]
        
        print("\n" + "="*60)
        print("🧪 TESTING CORRECTIVE RAG WITH SAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Test {i}: {question}")
            print("-" * 50)
            
            try:
                # Test using retrieve method
                answers = crag_searcher.retrieve(question)
                
                print(f"📤 Answer: {answers[0]}")
                
                # Get retrieved documents
                docs = crag_searcher.get_retrieved_documents()
                print(f"📚 Retrieved {len(docs)} documents")
                
                if docs:
                    print("📄 Sample document sources:")
                    for j, doc in enumerate(docs[:3]):  # Show first 3 docs
                        source = doc.metadata.get('source', 'Unknown')
                        preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"   {j+1}. {source}: {preview}...")
                
                print("✅ Test passed!")
                
            except Exception as e:
                print(f"❌ Test failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("🧪 TESTING INVOKE METHOD")
        print("="*60)
        
        # Test invoke method
        test_input = {"question": "What is artificial intelligence?"}
        print(f"📝 Testing invoke with: {test_input}")
        
        try:
            result = crag_searcher.invoke(test_input)
            print(f"📤 Answer: {result['answer']}")
            print(f"📚 Documents count: {len(result['documents'])}")
            print(f"🌐 Web search used: {result['web_search_used']}")
            print("✅ Invoke test passed!")
            
        except Exception as e:
            print(f"❌ Invoke test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("🎉 CORRECTIVE RAG TEST COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Failed to initialize Corrective RAG: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrective_rag() 