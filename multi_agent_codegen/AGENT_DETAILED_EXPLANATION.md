# Detailed Agent Explanations

This document provides in-depth explanations of each agent for thorough understanding.

---

## 🤖 Agent 1: Coder Agent

### Purpose
Generate Python code from natural language specifications.

### Detailed Workflow

#### 1. Input Processing
```python
specification = state.get("specification", "")
needs_improvement = state.get("needs_improvement", False)
previous_rating = state.get("code_rating", 0)
previous_feedback = state.get("rating_feedback", "")
```

#### 2. Mode Detection
- **Normal Mode**: First-time code generation
- **Improvement Mode**: Regenerating code based on feedback

#### 3. Prompt Construction

**Normal Mode Prompt**:
```
System: "You are an expert Python programmer. Generate clean, well-documented code..."

User: "Generate a Python function based on this specification:
[specification]

If the specification includes function signature or examples, follow them exactly.
Generate complete, production-ready code."
```

**Improvement Mode Prompt**:
```
System: [Same as normal]

User: "Generate a Python function based on this specification:
[specification]

IMPORTANT: The previous code received a rating of X/10. 
Feedback: [detailed feedback]
Generate significantly improved code that addresses all issues mentioned in the feedback.
Make sure the code is correct, handles all edge cases, follows best practices, and is well-documented."
```

#### 4. LLM Call
```python
generated_code = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.7,  # Balanced creativity
    max_tokens=2048
)
```

#### 5. Code Cleaning
- Removes markdown code blocks (```python ... ```)
- Removes REPL prompts (>>>)
- Extracts pure Python code
- Validates syntax

#### 6. State Update
```python
state["code"] = cleaned_code
state["code_history"].append(cleaned_code)
state["iteration"] += 1
state["last_agent"] = "coder"
```

### Key Features

**Improvement Mode**:
- Receives previous rating and feedback
- Uses feedback to guide generation
- Focuses on addressing specific issues
- Aims for higher quality code

**Code Quality**:
- Includes type hints
- Adds docstrings
- Handles edge cases
- Follows Python best practices

### Example

**Input**: "Write a function to find the maximum element in a list"

**Output**:
```python
def find_maximum(numbers: list) -> int:
    """
    Find the maximum element in a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Maximum number in the list
        
    Raises:
        ValueError: If list is empty
    """
    if not numbers:
        raise ValueError("List cannot be empty")
    return max(numbers)
```

---

## 🧪 Agent 2: Tester Agent

### Purpose
Generate comprehensive unit tests and execute them in a secure sandbox.

### Detailed Workflow

#### 1. Code Analysis
```python
code = state.get("code", "")
# Extract function names
# Parse function signatures
# Identify parameters
```

#### 2. Test Generation Prompt
```
System: "You are an expert at writing pytest unit tests. Generate comprehensive tests..."

User: "Generate pytest unit tests for this Python code:

Code:
[code]

CRITICAL REQUIREMENTS:
1. Use pytest framework
2. Import the function correctly
3. Test normal cases
4. Test edge cases (empty input, None, etc.)
5. Test boundary conditions
6. Return ONLY test code (no explanations, no markdown)
7. Use assert statements
8. Include descriptive test names"
```

#### 3. Test Code Cleaning
- Removes markdown blocks
- Ensures single import statement
- Removes duplicate function definitions
- Validates pytest syntax

#### 4. Syntax Validation
```python
# Validate generated code syntax
compile(code, '<string>', 'exec')

# Validate test code syntax
compile(test_code, '<string>', 'exec')
```

#### 5. Test Execution
```python
# Write code to temporary file
with open('temp_code.py', 'w') as f:
    f.write(code)

# Write tests to temporary file
with open('test_temp.py', 'w') as f:
    f.write(test_code)

# Execute tests
result = subprocess.run(
    ['pytest', 'test_temp.py', '--tb=long', '--no-header'],
    capture_output=True,
    text=True,
    timeout=30
)
```

#### 6. Result Parsing
```python
# Parse pytest output
output = result.stdout + result.stderr

# Count passed tests
passed = output.count('PASSED')

# Count failed tests
failed = output.count('FAILED')

# Extract errors
errors = extract_errors(output)

# Update state
state["tests_passed"] = (failed == 0)
state["test_results"] = {
    "passed": passed,
    "failed": failed,
    "total": passed + failed
}
state["test_output"] = output
state["test_errors"] = errors
```

### Key Features

**Comprehensive Testing**:
- Normal cases
- Edge cases (empty, None, single element)
- Boundary conditions
- Error cases

**Robust Execution**:
- Timeout protection (30 seconds)
- Error capture
- Verbose output for debugging
- Syntax validation

### Example

**Input Code**:
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
```

**Generated Tests**:
```python
import pytest
from temp_code import factorial

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_one():
    assert factorial(1) == 1

def test_factorial_positive():
    assert factorial(5) == 120
    assert factorial(3) == 6

def test_factorial_large():
    assert factorial(10) == 3628800
```

**Execution Result**:
- Tests: 4
- Passed: 4
- Failed: 0
- Status: PASS

---

## 🐛 Agent 3: Debugger Agent

### Purpose
Analyze test failures and generate targeted code fixes.

### Detailed Workflow

#### 1. Error Analysis
```python
test_output = state.get("test_output", "")
test_errors = state.get("test_errors", "")

# Extract error summary
error_summary = extract_error_summary(test_output)
# Identifies: Error type, location, message
```

#### 2. Error Summary Extraction
```python
def extract_error_summary(output):
    # Look for common error patterns
    if "SyntaxError" in output:
        return "SyntaxError: " + extract_message(output)
    elif "NameError" in output:
        return "NameError: " + extract_message(output)
    elif "AttributeError" in output:
        return "AttributeError: " + extract_message(output)
    # ... more error types
```

#### 3. Fix Generation Prompt
```
System: "You are an expert debugger. Fix Python code based on test failures..."

User: "Fix the following code based on test errors:

Code:
[code]

Test Code:
[test_code]

Errors:
[error_summary]

CRITICAL REQUIREMENTS:
1. Fix ALL errors mentioned
2. Maintain original function signatures
3. Return ONLY the fixed code (no explanations, no markdown)
4. Ensure code is correct and complete
5. Address all test failures
6. Keep the same function names and parameters"
```

#### 4. Code Fixing
```python
fixed_code = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.5,  # Lower for more focused fixes
    max_tokens=2048
)
```

#### 5. Code Cleaning
- Removes markdown
- Extracts pure Python
- Validates syntax

#### 6. State Update
```python
state["code"] = fixed_code
state["code_history"].append(fixed_code)
state["fix_history"].append({
    "iteration": state["iteration"],
    "error": error_summary,
    "fix": fixed_code
})
state["iteration"] += 1
```

### Key Features

**Error Analysis**:
- Identifies error types
- Extracts error messages
- Locates error positions
- Summarizes key issues

**Targeted Fixes**:
- Addresses specific errors
- Maintains function signatures
- Preserves code structure
- Fixes all errors in one pass

### Example

**Original Code** (with error):
```python
def add(a, b):
    return a + b
```

**Test Error**:
```
TypeError: add() missing 1 required positional argument: 'b'
```

**Fixed Code**:
```python
def add(a, b):
    if b is None:
        raise TypeError("add() missing 1 required positional argument: 'b'")
    return a + b
```

---

## ⭐ Agent 4: Rater Agent

### Purpose
Evaluate code quality and provide 0-10 rating with detailed feedback.

### Detailed Workflow

#### 1. Code Analysis
```python
code = state.get("code", "")
specification = state.get("specification", "")
```

#### 2. Rating Prompt
```
System: "You are an expert code reviewer. Evaluate Python code quality..."

User: "Rate the following Python code on a scale of 0-10:

Code:
[code]

Original Specification:
[specification]

Evaluate based on:
1. Correctness: Does it work correctly?
2. Completeness: Does it handle all cases?
3. Code Quality: Style, readability, structure
4. Best Practices: Python idioms, patterns
5. Efficiency: Algorithm complexity, performance
6. Documentation: Docstrings, comments

Provide:
1. Rating (0-10): [number]
2. Brief Explanation: [1-2 sentences]
3. Detailed Feedback: [paragraph with specific suggestions]"
```

#### 3. LLM Call
```python
response = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.3,  # Lower for consistency
    max_tokens=1024
)
```

#### 4. Response Parsing
```python
# Extract rating
rating = extract_rating(response)  # 0.0-10.0

# Extract details
details = extract_details(response)

# Extract feedback
feedback = extract_feedback(response)
```

#### 5. State Update
```python
state["code_rating"] = rating
state["rating_details"] = details
state["rating_feedback"] = feedback
state["iteration"] += 1
```

### Rating Scale

- **10**: Perfect, production-ready
- **9**: Excellent, minor improvements possible
- **8**: Very good, meets all requirements
- **7**: Good, acceptable quality
- **6**: Fair, needs some improvements
- **5**: Below average, significant issues
- **4**: Poor, many problems
- **3**: Very poor, major flaws
- **2**: Bad, mostly incorrect
- **1**: Very bad, doesn't work
- **0**: Completely broken

### Example

**Code**:
```python
def reverse_string(s):
    return s[::-1]
```

**Rating Response**:
```
Rating: 7.5/10

Brief Explanation: Simple and correct implementation, but lacks type hints and error handling.

Detailed Feedback: The code correctly reverses a string using Python slicing, which is efficient. However, it would benefit from:
1. Type hints: def reverse_string(s: str) -> str
2. Error handling: Check if input is a string
3. Documentation: Add a docstring explaining the function
4. Edge cases: Handle None or empty string explicitly
```

---

## 🚀 Agent 5: Optimizer Agent

### Purpose
Refine code efficiency and add comprehensive documentation.

### Detailed Workflow

#### 1. Code Receipt
- Only runs if `code_rating >= 7.0`
- Receives high-quality code

#### 2. Optimization Prompt
```
System: "You are an expert code optimizer. Improve Python code quality..."

User: "Optimize and document the following code:

Code:
[code]

Add:
1. Comprehensive docstrings (Google style)
2. Type hints
3. Performance optimizations (if applicable)
4. Better variable names (if needed)
5. Comments for complex logic
6. Error handling improvements

Return the optimized code with all improvements."
```

#### 3. Optimization
```python
optimized_code = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.3,  # Lower for focused optimization
    max_tokens=1024
)
```

#### 4. State Update
```python
state["code"] = optimized_code
state["optimized"] = True
state["iteration"] += 1
```

### Optimizations Applied

1. **Documentation**:
   - Google-style docstrings
   - Parameter descriptions
   - Return value descriptions
   - Example usage

2. **Type Hints**:
   - Function parameters
   - Return types
   - Variable types (where helpful)

3. **Performance**:
   - Algorithm improvements
   - Efficient data structures
   - Reduced complexity

4. **Readability**:
   - Better variable names
   - Code structure
   - Comments for complex logic

### Example

**Before**:
```python
def fact(n):
    if n == 0:
        return 1
    return n * fact(n-1)
```

**After**:
```python
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.
    
    Uses recursive approach to compute n!.
    
    Args:
        n: Non-negative integer to compute factorial for
        
    Returns:
        Factorial of n (n!)
        
    Raises:
        ValueError: If n is negative
        
    Example:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

---

## 💬 Agent 6: Explainer Agent

### Purpose
Answer questions about generated code.

### Detailed Workflow

#### 1. Question Receipt
```python
code = request.json.get("code", "")
question = request.json.get("question", "")
specification = request.json.get("specification", "")
```

#### 2. Explanation Prompt
```
System: "You are an expert code explainer. Answer questions about Python code clearly..."

User: "Answer this question about the following Python code:

Code:
[code]

Original specification:
[specification]

Question: [question]

Provide a clear, helpful answer. If the question is about a specific part of the code, reference it specifically."
```

#### 3. Generation
```python
answer = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    temperature=0.7,  # Higher for creative explanations
    max_tokens=2048
)
```

#### 4. Response
```json
{
    "answer": "The code works by...",
    "question": "How does this code work?"
}
```

### Example Questions & Answers

**Q**: "How does this code work?"
**A**: "This code implements a binary search algorithm. It works by repeatedly dividing the search space in half. First, it checks if the target is at the middle position. If not, it eliminates half of the remaining elements and continues searching in the remaining half. This continues until the target is found or the search space is exhausted. The time complexity is O(log n), making it very efficient for large sorted arrays."

**Q**: "What are the edge cases?"
**A**: "The code handles several edge cases: 1) Empty array - returns -1 immediately, 2) Single element array - checks if it matches the target, 3) Target not found - returns -1 after exhausting search space, 4) Target at boundaries - handles first and last elements correctly."

---

## 🔄 Agent Interaction

### State Flow

```
State starts with: {specification: "..."}

Coder → {code: "...", code_history: [...]}
Tester → {tests_passed: True/False, test_results: {...}}
Debugger → {code: "fixed...", fix_history: [...]}
Rater → {code_rating: 8.5, rating_feedback: "..."}
Optimizer → {code: "optimized...", optimized: True}
Explainer → {answer: "...", question: "..."}
```

### Error Propagation

- Each agent catches errors
- Errors stored in `state["error"]`
- Next agent can handle or propagate
- Final error shown to user

### Iteration Control

- `iteration` counter tracks cycles
- `max_iterations` limits total cycles
- `improvement_attempts` tracks improvement loops
- System stops when conditions met

---

**This completes the detailed agent explanations. Each agent is designed to work independently while contributing to the overall goal of generating high-quality code.**

