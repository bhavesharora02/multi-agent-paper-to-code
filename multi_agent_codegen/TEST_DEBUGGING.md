# Test Failure Debugging Guide

## Common Reasons for Test Failures

### 1. **Code Generation Issues**
- Generated code might have syntax errors
- Function names might not match test expectations
- Code might not handle edge cases properly

### 2. **Test Generation Issues**
- Tests might import functions incorrectly
- Test assertions might be too strict or incorrect
- Tests might not match the actual function signature

### 3. **Import Problems**
- Functions might not be imported correctly
- Module path issues in test execution

### 4. **Logic Errors**
- Generated code might have logical errors
- Edge cases might not be handled

## How to Debug

### Check the Test Output
The system now provides detailed error messages. Look for:
- Syntax errors in generated code
- Import errors
- Assertion failures
- Function signature mismatches

### View Detailed Errors
In the web interface, check the "Test Output" section to see:
- Full pytest output
- Error tracebacks
- Which specific tests failed

### Common Fixes

1. **If tests can't import functions:**
   - The debugger agent should fix this automatically
   - Check that function names match between code and tests

2. **If assertions fail:**
   - The debugger agent will analyze errors and fix the code
   - Check if test expectations are too strict

3. **If syntax errors occur:**
   - The code extraction might have issues
   - The debugger should fix syntax errors

## Improvements Made

### Enhanced Test Generation
- Better prompts for test generation
- Automatic function name extraction
- Proper import statements
- Cleaner test code

### Better Error Handling
- Syntax validation before testing
- More detailed error messages
- Better pytest output parsing

### Code Cleaning
- Removes markdown formatting
- Cleans up generated code
- Validates syntax before execution

## What to Expect

### First Iteration
- Coder generates initial code
- Tester generates and runs tests
- If tests fail, Debugger analyzes and fixes

### Subsequent Iterations
- Debugger fixes issues
- Tester re-runs tests
- Process continues until tests pass or max iterations

### Success Indicators
- "Tests Passed" status
- Green checkmark
- Generated code available for download

## Tips

1. **Be specific in specifications:**
   - Include expected behavior
   - Mention edge cases
   - Provide examples if possible

2. **Check max iterations:**
   - Increase if complex problems
   - Default is 10 iterations

3. **Review generated code:**
   - Check if it matches your specification
   - Look for obvious errors

4. **Check test output:**
   - See what specific tests failed
   - Understand why they failed

## Still Having Issues?

If tests continue to fail after multiple iterations:
1. Check the test output for specific errors
2. Review the generated code
3. Try a simpler specification first
4. Check that pytest is installed: `pip install pytest`

