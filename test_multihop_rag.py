import json
import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load API keys from .env
from dotenv import load_dotenv
load_dotenv()

# Import necessary modules
from searchers.corrective_rag import CorrectiveRAGSearcher
from chat_models.openai_chat_model import OpenAIChatModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test questions from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'questions' in data:
            return data['questions']
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            # Try to extract questions from the structure
            questions = []
            for key, value in data.items():
                if isinstance(value, dict) and 'question' in value:
                    questions.append(value)
            return questions
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def test_corrective_rag_batch(
    test_file: str,
    max_questions: int = 10,
    output_file: str = "test_results.json"
):
    """Test Corrective RAG on batch of questions."""
    print("🚀 Starting Batch Test for Corrective RAG...")
    
    # Initialize components
    print("📋 Initializing components...")
    
    # LLM
    llm = OpenAIChatModel(model_name="gpt-4o-mini", temperature=0)
    print("✅ LLM initialized (GPT-4o Mini)")
    
    # Embeddings
    embeddings = OpenAIEmbeddings()
    print("✅ Embeddings initialized")
    
    # Web search tool
    web_search_tool = TavilySearchResults(k=3)
    print("✅ Web search tool initialized")
    
    # FAISS index path and corpus path
    faiss_index_path = "data/faiss_index"
    corpus_path = "data/multihoprag_corpus.txt"
    
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
    
    # Load test data
    print(f"📂 Loading test data from: {test_file}")
    test_questions = load_test_data(test_file)
    
    if not test_questions:
        print("❌ No test questions found!")
        return
    
    print(f"📚 Found {len(test_questions)} questions")
    
    # Limit questions for testing
    test_questions = test_questions[:max_questions]
    print(f"🎯 Testing first {len(test_questions)} questions")
    
    # Results storage
    results = []
    
    print("\n" + "="*80)
    print("🧪 STARTING BATCH TEST")
    print("="*80)
    
    for i, question_data in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}/{len(test_questions)}")
        print("-" * 60)
        
        # Extract question text (handle different formats)
        question = ""
        expected_answer = ""
        
        if isinstance(question_data, str):
            question = question_data
        elif isinstance(question_data, dict):
            # Try different possible keys
            question = (question_data.get('question') or 
                       question_data.get('query') or 
                       question_data.get('text') or
                       str(question_data))
            expected_answer = (question_data.get('answer') or 
                             question_data.get('expected') or 
                             question_data.get('ground_truth') or "")
        
        if not question:
            print(f"⚠️ Could not extract question from item {i}")
            continue
            
        print(f"❓ Question: {question}")
        
        try:
            # Record start time
            start_time = time.time()
            
            # Test using invoke method
            result = crag_searcher.invoke({"question": question})
            
            # Record end time
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract results
            answer = result['answer']
            documents_count = len(result['documents'])
            web_search_used = result['web_search_used']
            
            print(f"📤 Answer: {answer[:200]}..." if len(answer) > 200 else f"📤 Answer: {answer}")
            print(f"📚 Documents used: {documents_count}")
            print(f"🌐 Web search used: {web_search_used}")
            print(f"⏱️ Response time: {response_time:.2f}s")
            
            # Store result
            result_data = {
                "question_id": i,
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "documents_count": documents_count,
                "web_search_used": web_search_used,
                "response_time": response_time,
                "status": "success"
            }
            
            results.append(result_data)
            print("✅ Test passed!")
            
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            result_data = {
                "question_id": i,
                "question": question,
                "answer": "",
                "expected_answer": expected_answer,
                "documents_count": 0,
                "web_search_used": False,
                "response_time": 0,
                "status": "failed",
                "error": str(e)
            }
            results.append(result_data)
    
    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "total_questions": len(test_questions),
                "successful": len([r for r in results if r['status'] == 'success']),
                "failed": len([r for r in results if r['status'] == 'failed']),
                "avg_response_time": sum(r['response_time'] for r in results if r['status'] == 'success') / len([r for r in results if r['status'] == 'success']) if results else 0,
                "web_search_usage": len([r for r in results if r.get('web_search_used', False)]) / len(results) * 100 if results else 0
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    web_used = len([r for r in results if r.get('web_search_used', False)])
    
    print(f"✅ Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Failed: {failed}/{len(results)} ({failed/len(results)*100:.1f}%)")
    print(f"🌐 Web search used: {web_used}/{len(results)} ({web_used/len(results)*100:.1f}%)")
    
    if successful > 0:
        avg_time = sum(r['response_time'] for r in results if r['status'] == 'success') / successful
        print(f"⏱️ Average response time: {avg_time:.2f}s")
    
    print(f"💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    # Path to test file
    test_file_path = "../MultiHopRAG (1).json"
    
    # Check if file exists
    if not os.path.exists(test_file_path):
        print(f"❌ Test file not found: {test_file_path}")
        print("Please make sure the MultiHopRAG (1).json file is in the correct location.")
        sys.exit(1)
    
    # Run batch test
    test_corrective_rag_batch(
        test_file=test_file_path,
        max_questions=5,  # Start with 5 questions for testing
        output_file="multihop_test_results.json"
    ) 