#!/usr/bin/env python3
"""
Accuracy validation script for Ministral-8B on Tenstorrent N300 hardware.
Tests model outputs against expected responses and validates model functionality.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple
import time

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Test questions and expected response patterns
ACCURACY_TESTS = [
    {
        "question": "What is the capital of France?",
        "expected_keywords": ["Paris", "paris", "France", "france"],
        "category": "factual",
        "max_length": 50
    },
    {
        "question": "Explain what artificial intelligence is in one sentence.",
        "expected_keywords": ["artificial", "intelligence", "AI", "machine", "computer", "learn"],
        "category": "explanation",
        "max_length": 100
    },
    {
        "question": "Write a simple Python function to add two numbers.",
        "expected_keywords": ["def", "return", "+", "function", "python"],
        "category": "coding",
        "max_length": 200
    },
    {
        "question": "What are the primary colors?",
        "expected_keywords": ["red", "blue", "yellow", "green", "primary"],
        "category": "factual",
        "max_length": 50
    },
    {
        "question": "How do you say 'hello' in Spanish?",
        "expected_keywords": ["hola", "Spanish", "spanish"],
        "category": "language",
        "max_length": 30
    },
    {
        "question": "What is 15 + 27?",
        "expected_keywords": ["42", "forty-two", "forty two"],
        "category": "math",
        "max_length": 20
    },
    {
        "question": "Name three planets in our solar system.",
        "expected_keywords": ["Earth", "Mars", "Venus", "Jupiter", "Saturn", "Mercury", "Neptune", "Uranus"],
        "category": "science",
        "max_length": 50
    },
    {
        "question": "What is the largest mammal on Earth?",
        "expected_keywords": ["whale", "blue whale", "largest", "mammal"],
        "category": "science",
        "max_length": 50
    }
]

def run_model_inference(question: str, device_id: int = 0, max_seq_len: int = 512) -> Tuple[str, bool]:
    """Run inference on the model and return the response."""
    try:
        # Create a temporary file for the question
        temp_file = "/tmp/test_question.txt"
        with open(temp_file, 'w') as f:
            f.write(question)
        
        # Run the model
        cmd = (f"cd /workspaces/tt-metal/models/demos/wormhole/ministral8b && "
               f"python demo/demo_with_prefill.py "
               f"--batch_size 1 --max_seq_len {max_seq_len} "
               f"--device_id {device_id} --instruct "
               f"--question '{question}' 2>/dev/null")
        
        # Capture output
        output_file = "/tmp/model_output.txt"
        result = os.system(f"{cmd} > {output_file}")
        
        if result == 0:
            with open(output_file, 'r') as f:
                response = f.read().strip()
            return response, True
        else:
            return f"Error: Command failed with code {result}", False
            
    except Exception as e:
        return f"Error: {str(e)}", False
    finally:
        # Clean up temp files
        for temp_file in ["/tmp/test_question.txt", "/tmp/model_output.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def evaluate_response(response: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a model response against expected criteria."""
    result = {
        "question": test_case["question"],
        "response": response,
        "category": test_case["category"],
        "score": 0.0,
        "details": {}
    }
    
    # Check if response is not empty
    if not response or response.startswith("Error:"):
        result["details"]["empty_response"] = True
        return result
    
    result["details"]["empty_response"] = False
    
    # Check response length
    response_length = len(response)
    max_length = test_case["max_length"]
    result["details"]["length"] = response_length
    result["details"]["within_length_limit"] = response_length <= max_length * 2  # Allow some flexibility
    
    # Check for expected keywords
    response_lower = response.lower()
    expected_keywords = test_case["expected_keywords"]
    found_keywords = []
    
    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
    
    result["details"]["found_keywords"] = found_keywords
    result["details"]["keyword_coverage"] = len(found_keywords) / len(expected_keywords)
    
    # Check response relevance (basic heuristics)
    result["details"]["contains_question_words"] = any(
        word in response_lower for word in test_case["question"].lower().split()[:3]
    )
    
    # Calculate overall score
    score = 0.0
    
    # Base score for non-empty response
    if not result["details"]["empty_response"]:
        score += 0.3
    
    # Score for keyword coverage
    score += result["details"]["keyword_coverage"] * 0.5
    
    # Score for reasonable length
    if result["details"]["within_length_limit"]:
        score += 0.1
    
    # Score for relevance
    if result["details"]["contains_question_words"]:
        score += 0.1
    
    result["score"] = min(score, 1.0)
    
    return result

def check_hardware():
    """Check if N300 hardware is available and detected."""
    try:
        import ttnn
        devices = ttnn.get_device_ids()
        print(f"✅ Found {len(devices)} Tenstorrent device(s): {devices}")
        return len(devices) > 0
    except Exception as e:
        print(f"❌ Error accessing Tenstorrent devices: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate accuracy of Ministral-8B on N300 hardware")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--output_file", type=str, default="accuracy_results.json", help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer questions")
    parser.add_argument("--verbose", action="store_true", help="Print detailed responses")
    
    args = parser.parse_args()
    
    print("🎯 Starting Ministral-8B Accuracy Validation on N300 Hardware")
    print("=" * 60)
    
    # Check hardware
    if not check_hardware():
        print("❌ No Tenstorrent hardware detected. Exiting.")
        return 1
    
    # Select test cases
    test_cases = ACCURACY_TESTS[:4] if args.quick else ACCURACY_TESTS
    
    print(f"\n📝 Running {len(test_cases)} accuracy tests...")
    
    results = []
    category_scores = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Testing: {test_case['category']} question")
        print(f"Q: {test_case['question']}")
        
        # Run inference
        start_time = time.time()
        response, success = run_model_inference(
            test_case["question"], 
            args.device_id, 
            args.max_seq_len
        )
        inference_time = time.time() - start_time
        
        if not success:
            print(f"❌ Inference failed: {response}")
            result = {
                "question": test_case["question"],
                "response": response,
                "category": test_case["category"],
                "score": 0.0,
                "inference_time": inference_time,
                "success": False
            }
        else:
            # Evaluate response
            result = evaluate_response(response, test_case)
            result["inference_time"] = inference_time
            result["success"] = True
            
            print(f"A: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"📊 Score: {result['score']:.2f} ({inference_time:.2f}s)")
            
            if args.verbose:
                print(f"   Keywords found: {result['details']['found_keywords']}")
                print(f"   Keyword coverage: {result['details']['keyword_coverage']:.1%}")
        
        results.append(result)
        
        # Track category scores
        category = test_case["category"]
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(result["score"])
    
    # Calculate summary statistics
    successful_tests = [r for r in results if r["success"]]
    overall_score = sum(r["score"] for r in results) / len(results)
    success_rate = len(successful_tests) / len(results)
    avg_inference_time = sum(r["inference_time"] for r in results) / len(results)
    
    # Save results
    summary = {
        "overall_score": overall_score,
        "success_rate": success_rate,
        "avg_inference_time": avg_inference_time,
        "total_tests": len(results),
        "successful_tests": len(successful_tests),
        "category_scores": {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()},
        "individual_results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 ACCURACY VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"✅ Overall Score: {overall_score:.1%}")
    print(f"✅ Success Rate: {success_rate:.1%} ({len(successful_tests)}/{len(results)} tests)")
    print(f"⏱️  Average Inference Time: {avg_inference_time:.2f}s")
    
    print("\n📊 CATEGORY SCORES:")
    for category, avg_score in summary["category_scores"].items():
        print(f"  {category.capitalize()}: {avg_score:.1%}")
    
    print("\n🎯 VALIDATION TARGETS:")
    
    # Check if we meet minimum thresholds
    targets = {
        "Overall accuracy": (overall_score >= 0.7, f"{overall_score:.1%}"),
        "Success rate": (success_rate >= 0.9, f"{success_rate:.1%}"),
        "Avg inference time": (avg_inference_time <= 10.0, f"{avg_inference_time:.2f}s")
    }
    
    for target_name, (passed, value) in targets.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {target_name}: {value} {status}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if overall_score < 0.7:
        print("  - Model accuracy is below target (70%). Consider fine-tuning or adjusting inference parameters.")
    if success_rate < 0.9:
        print("  - Some tests failed to run. Check hardware stability and model loading.")
    if avg_inference_time > 10.0:
        print("  - Inference time is high. Consider optimizing batch size or sequence length.")
    
    if overall_score >= 0.7 and success_rate >= 0.9 and avg_inference_time <= 10.0:
        print("  🎉 All targets met! Model is ready for production.")
    
    print(f"\n💾 Full results saved to: {args.output_file}")
    
    # Return appropriate exit code
    return 0 if (overall_score >= 0.7 and success_rate >= 0.9) else 1

if __name__ == "__main__":
    exit(main())
